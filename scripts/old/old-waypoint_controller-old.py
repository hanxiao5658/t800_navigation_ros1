#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 ordered-waypoint navigator (ROS1 Noetic).

Pose:     /my_ros1/navigation/base_pose_gt (geometry_msgs/PoseStamped)
Velocity: /my_ros1/motion/body_vel_cmd   (geometry_msgs/Twist, BODY frame)
Status:   /my_ros1/navigation/state       (std_msgs/String)
Goal:     /my_ros1/navigation/active_waypoint (geometry_msgs/PoseStamped)
Path:     /my_ros1/navigation/waypoints_path  (nav_msgs/Path)
Services: /my_ros1/navigation/start, /my_ros1/navigation/pause (std_srvs/Trigger)

Waypoint yaw means desired FINAL orientation at the point. This version does NOT
implement obstacle avoidance/detection; add that as a separate stage.
"""

import math
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def saturated(value, limit):
    return max(-limit, min(limit, value))


def signed_min_magnitude(value, minimum):
    if value == 0.0:
        return 0.0
    return math.copysign(max(abs(value), minimum), value)


class WaypointController:
    def __init__(self):
        rospy.init_node('waypoint_controller')
        self.frame_id = rospy.get_param('~frame_id', 'world')
        self.pose_topic = rospy.get_param('~pose_topic', '/my_ros1/navigation/base_pose_gt')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/my_ros1/motion/body_vel_cmd')
        self.rate = float(rospy.get_param('~control_rate', 30.0))
        self.pose_timeout = float(rospy.get_param('~pose_timeout', 0.5))
        self.xy_tol = float(rospy.get_param('~xy_tolerance', 0.15))
        self.yaw_tol = math.radians(float(rospy.get_param('~yaw_tolerance_deg', 5.0)))
        self.realign = math.radians(float(rospy.get_param('~realign_threshold_deg', 12.0)))
        self.kp_forward = float(rospy.get_param('~kp_forward', 0.5))
        self.kp_turn = float(rospy.get_param('~kp_turn', 1.2))
        self.kp_steer = float(rospy.get_param('~kp_steer', 0.8))
        self.max_forward = float(rospy.get_param('~max_forward_speed', 0.20))
        self.min_forward = float(rospy.get_param('~min_forward_speed', 0.06))
        self.max_turn = float(rospy.get_param('~max_turn_speed', 0.35))
        self.min_turn = float(rospy.get_param('~min_turn_speed', 0.08))
        self.max_steer = float(rospy.get_param('~max_steering_speed', 0.12))
        self.autostart = bool(rospy.get_param('~autostart', False))
        if self.rate <= 0 or self.pose_timeout <= 0:
            raise ValueError('control_rate and pose_timeout must be > 0')
        if not (0.0 < self.yaw_tol < self.realign < math.pi):
            raise ValueError('yaw_tolerance_deg must be < realign_threshold_deg < 180')
        if self.xy_tol <= 0 or self.min_forward < 0 or self.max_forward <= 0:
            raise ValueError('Invalid translation tolerance/speed')
        if self.min_forward > self.max_forward or self.min_turn > self.max_turn:
            raise ValueError('Minimum speed cannot exceed maximum speed')
        if min(self.kp_forward, self.kp_turn, self.kp_steer) < 0:
            raise ValueError('P gains must be nonnegative')

        raw = rospy.get_param('~waypoints', [])
        if not isinstance(raw, list) or not raw:
            raise ValueError('~waypoints must be a nonempty YAML list')
        self.waypoints = []
        for i, entry in enumerate(raw):
            try:
                point = {
                    'id': str(entry.get('id', 'wp{}'.format(i + 1))),
                    'x': float(entry['x']),
                    'y': float(entry['y']),
                    'yaw': float(entry['yaw']),
                    'wait': float(entry.get('wait', 0.0))
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Invalid waypoint {}: {}'.format(i, exc))
            if not all(math.isfinite(point[k]) for k in ('x', 'y', 'yaw', 'wait')):
                raise ValueError('Non-finite waypoint {}'.format(i))
            if point['wait'] < 0:
                raise ValueError('wait must be >= 0')
            self.waypoints.append(point)

        self.lock = threading.Lock()
        self.pose = None  # (x,y,yaw) in Gazebo world
        self.pose_wall_time = None
        self.enabled = self.autostart
        self.index = 0
        self.state = 'WAIT_FOR_POSE' if self.autostart else 'IDLE'
        self.dwell_until = None

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            '/my_ros1/navigation/state', String, queue_size=5, latch=True)
        self.goal_pub = rospy.Publisher(
            '/my_ros1/navigation/active_waypoint', PoseStamped,
            queue_size=1, latch=True)
        self.path_pub = rospy.Publisher(
            '/my_ros1/navigation/waypoints_path', Path, queue_size=1, latch=True)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=1)
        rospy.Service('/my_ros1/navigation/start', Trigger, self.on_start)
        rospy.Service('/my_ros1/navigation/pause', Trigger, self.on_pause)
        rospy.on_shutdown(self.on_shutdown)
        self.publish_path()
        self.publish_goal()
        self.publish_status()
        rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self.control_step)
        rospy.loginfo('Waypoint navigator ready (%d waypoints; autostart=%s)',
                      len(self.waypoints), self.autostart)

    def on_pose(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.logwarn_throttle(5.0, 'Pose frame %r differs from %r',
                                   msg.header.frame_id, self.frame_id)
            return
        q = msg.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        p = msg.pose.position
        if norm < 1e-8 or not all(math.isfinite(v) for v in
                                  (p.x, p.y, q.x, q.y, q.z, q.w)):
            rospy.logwarn_throttle(5.0, 'Invalid ground-truth pose ignored')
            return
        yaw = euler_from_quaternion((q.x/norm, q.y/norm, q.z/norm, q.w/norm))[2]
        with self.lock:
            self.pose = (p.x, p.y, yaw)
            self.pose_wall_time = time.monotonic()

    def pose_is_fresh(self):
        with self.lock:
            return (self.pose is not None and self.pose_wall_time is not None
                    and time.monotonic() - self.pose_wall_time <= self.pose_timeout)

    def on_start(self, _req):
        if not self.pose_is_fresh():
            return TriggerResponse(success=False,
                                   message='No fresh ground-truth pose')
        if self.index >= len(self.waypoints):
            self.index = 0  # restart route if already completed
        self.enabled = True
        self.dwell_until = None
        self.change_state('ALIGN_HEADING')
        self.publish_goal()
        return TriggerResponse(success=True,
                               message='Navigation started at waypoint {}'.format(self.index + 1))

    def on_pause(self, _req):
        self.enabled = False
        self.dwell_until = None
        self.change_state('IDLE')
        self.publish_cmd(0.0, 0.0)
        return TriggerResponse(success=True, message='Navigation paused; zero velocity requested')

    def change_state(self, state):
        if self.state != state:
            self.state = state
            self.publish_status()
            rospy.loginfo('Nav state => %s; waypoint %d/%d',
                          state, min(self.index + 1, len(self.waypoints)),
                          len(self.waypoints))

    def publish_status(self):
        label = self.waypoints[min(self.index, len(self.waypoints)-1)]['id']
        self.status_pub.publish(String(data='state={} waypoint={}/{} id={}'.format(
            self.state, min(self.index + 1, len(self.waypoints)),
            len(self.waypoints), label)))

    def as_pose(self, wp):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = wp['x']
        msg.pose.position.y = wp['y']
        msg.pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, wp['yaw'])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg

    def publish_goal(self):
        if self.index < len(self.waypoints):
            self.goal_pub.publish(self.as_pose(self.waypoints[self.index]))

    def publish_path(self):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = rospy.Time.now()
        path.poses = [self.as_pose(p) for p in self.waypoints]
        self.path_pub.publish(path)

    def publish_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)  # +x: ROBOT forward, not world x
        msg.angular.z = float(wz)  # +z: robot CCW
        self.cmd_pub.publish(msg)

    def turn_cmd(self, err):
        raw = saturated(self.kp_turn * err, self.max_turn)
        return saturated(signed_min_magnitude(raw, self.min_turn), self.max_turn)

    def advance(self):
        self.index += 1
        self.dwell_until = None
        if self.index >= len(self.waypoints):
            self.enabled = False
            self.change_state('FINISHED')
            rospy.loginfo('Completed all %d waypoints', len(self.waypoints))
        else:
            self.publish_goal()
            self.change_state('ALIGN_HEADING')

    def control_step(self, _event):
        if not self.enabled:
            self.publish_cmd(0.0, 0.0)
            return
        if not self.pose_is_fresh():
            self.change_state('WAIT_FOR_POSE')
            self.publish_cmd(0.0, 0.0)
            return
        if self.state == 'WAIT_FOR_POSE':
            self.change_state('ALIGN_HEADING')
        with self.lock:
            x, y, yaw = self.pose
        goal = self.waypoints[self.index]
        dx = goal['x'] - x
        dy = goal['y'] - y
        dist = math.hypot(dx, dy)

        if self.state == 'ALIGN_HEADING':
            if dist <= self.xy_tol:
                self.change_state('ALIGN_FINAL_YAW')
                self.publish_cmd(0.0, 0.0)
                return
            err = wrap_pi(math.atan2(dy, dx) - yaw)
            if abs(err) <= self.yaw_tol:
                self.change_state('MOVE_FORWARD')
                self.publish_cmd(0.0, 0.0)
            else:
                self.publish_cmd(0.0, self.turn_cmd(err))

        elif self.state == 'MOVE_FORWARD':
            if dist <= self.xy_tol:
                self.change_state('ALIGN_FINAL_YAW')
                self.publish_cmd(0.0, 0.0)
                return
            err = wrap_pi(math.atan2(dy, dx) - yaw)
            if abs(err) >= self.realign:
                self.change_state('ALIGN_HEADING')
                self.publish_cmd(0.0, 0.0)
                return
            vx = min(self.max_forward,
                     max(self.min_forward, self.kp_forward * dist))
            wz = saturated(self.kp_steer * err, self.max_steer)
            self.publish_cmd(vx, wz)

        elif self.state == 'ALIGN_FINAL_YAW':
            err = wrap_pi(goal['yaw'] - yaw)
            if abs(err) <= self.yaw_tol:
                self.dwell_until = rospy.Time.now().to_sec() + goal['wait']
                self.change_state('DWELL')
                self.publish_cmd(0.0, 0.0)
            else:
                self.publish_cmd(0.0, self.turn_cmd(err))

        elif self.state == 'DWELL':
            self.publish_cmd(0.0, 0.0)
            if rospy.Time.now().to_sec() >= self.dwell_until:
                self.advance()
        else:
            self.publish_cmd(0.0, 0.0)

    def on_shutdown(self):
        # Navigation is responsible for stopping, because downstream adapters
        # intentionally do not have a command watchdog.
        for _ in range(5):
            try:
                self.publish_cmd(0.0, 0.0)
                time.sleep(0.02)
            except Exception:
                break


if __name__ == '__main__':
    try:
        WaypointController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
