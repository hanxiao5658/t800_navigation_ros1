#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 ROS1 Noetic 顺序航点导航控制器（仿真/实机共用）。

状态流程：
  /start -> ALIGN_HEADING -> MOVE_FORWARD -> ALIGN_FINAL_YAW
         -> WAIT_FOR_NEXT (需人工 /next_waypoint) -> ... -> FINISHED
  障碍阻挡/数据超时：停车；障碍恢复后保持原有的自动恢复行为。
  定位状态、位姿或时间戳异常：LOCALIZATION_FAULT_LOCKED + 停车；
      数据恢复后仍锁定，必须人工重新调用 /start 才可继续。

输入：
  ~pose_topic                         geometry_msgs/PoseStamped
  ~obstacle_topic                     std_msgs/String  CLEAR|goal=N|...
  ~gicp_status_topic                  std_msgs/String  ALIGNED_INITIAL_ONLY
  ~pose_status_topic                  std_msgs/String  TRACKING_FIXED_GICP

输出：
  /my_ros1/motion/body_vel_cmd         geometry_msgs/Twist
  /my_ros1/navigation/state           std_msgs/String (latched)
  /my_ros1/navigation/active_waypoint geometry_msgs/PoseStamped (latched)
  /my_ros1/navigation/obstacle_target std_msgs/String (latched)
  /my_ros1/navigation/waypoints_path  nav_msgs/Path (latched)
  /my_ros1/navigation/controller_heartbeat  std_msgs/String (NOT latched)

服务：/my_ros1/navigation/start, /pause, /next_waypoint (std_srvs/Trigger)

重要安全边界：
  * heartbeat 由控制回调发布，不是独立于控制回调的伪存活线程；
    ROS2 下游必须用自己的 monotonic 接收时间检测 heartbeat 和 Twist 超时，
    超时或 motion_allowed=0 时主动置零/禁用运动，不能只依赖 ROS1 停车命令。
  * 这不是经过认证的功能安全/急停机制，独立硬件急停仍然必需。
  * localization_enabled 必须在实机启用，仿真没有定位话题时可设 false。
  * 脚本不自动做 TF 坐标转换。waypoints.frame_id 必须与 pose.header.frame_id 相同。
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
        p = rospy.get_param

        # 原有参数/控制律保持兼容：通过 waypoints_{real,sim}.yaml 加载 frame_id。
        self.frame_id = p('~frame_id', 'world')
        self.pose_topic = p('~pose_topic', '/my_ros1/navigation/base_pose_gt')
        self.cmd_topic = p('~cmd_topic', '/my_ros1/motion/body_vel_cmd')
        self.rate = float(p('~control_rate', 30.0))
        self.pose_timeout = float(p('~pose_timeout', 0.5))
        self.xy_tol = float(p('~xy_tolerance', 0.15))
        self.yaw_tol = math.radians(float(p('~yaw_tolerance_deg', 5.0)))
        self.realign = math.radians(float(p('~realign_threshold_deg', 12.0)))
        self.kp_forward = float(p('~kp_forward', 0.5))
        self.kp_turn = float(p('~kp_turn', 1.2))
        self.kp_steer = float(p('~kp_steer', 0.8))
        self.max_forward = float(p('~max_forward_speed', 0.20))
        self.min_forward = float(p('~min_forward_speed', 0.06))
        self.max_turn = float(p('~max_turn_speed', 0.35))
        self.min_turn = float(p('~min_turn_speed', 0.08))
        self.max_steer = float(p('~max_steering_speed', 0.12))
        self.autostart = bool(p('~autostart', False))

        self.obstacle_enabled = bool(p('~obstacle_enabled', True))
        self.obstacle_topic = p('~obstacle_topic', '/my_ros1/navigation/obstacle_status')
        self.obstacle_timeout = float(p('~obstacle_timeout', 0.5))
        self.safety_poll_rate = float(p('~safety_poll_rate', 20.0))

        # 默认 FAIL-CLOSED：实机必须启用定位门控。
        # 仿真可在 navigation_params_sim.yaml 显式设置 localization_enabled: false。
        self.localization_enabled = bool(p('~localization_enabled', True))
        self.gicp_status_topic = p('~gicp_status_topic', '/my_ros1/localization/status')
        self.pose_status_topic = p('~pose_status_topic', '/my_ros1/localization/pose_status')
        # FastGICP status 约 1 Hz、pose_status 约 2 Hz。
        self.gicp_status_timeout = float(p('~gicp_status_timeout', 3.0))
        self.pose_status_timeout = float(p('~pose_status_timeout', 2.0))
        self.pose_future_tolerance = float(p('~pose_future_tolerance', 0.1))
        # 心跳在控制周期发布；下游应自行以接收时间测超时。
        self.heartbeat_topic = p('~heartbeat_topic', '/my_ros1/navigation/controller_heartbeat')

        if not self.frame_id or not self.heartbeat_topic:
            raise ValueError('frame_id and heartbeat_topic must not be empty')
        if self.rate <= 0 or self.pose_timeout <= 0 or self.safety_poll_rate <= 0:
            raise ValueError('control_rate, pose_timeout, safety_poll_rate must be > 0')
        if self.obstacle_timeout <= 0 or self.gicp_status_timeout <= 0 or self.pose_status_timeout <= 0:
            raise ValueError('status timeouts must be > 0')
        if self.pose_future_tolerance < 0:
            raise ValueError('pose_future_tolerance must be >= 0')
        if not (0 < self.yaw_tol < self.realign < math.pi):
            raise ValueError('yaw_tolerance_deg must be < realign_threshold_deg < 180')
        if self.xy_tol <= 0 or self.min_forward < 0 or self.max_forward <= 0:
            raise ValueError('Invalid translation tolerance/speed')
        if self.min_forward > self.max_forward or self.min_turn > self.max_turn:
            raise ValueError('Minimum speed cannot exceed maximum speed')
        if min(self.kp_forward, self.kp_turn, self.kp_steer) < 0:
            raise ValueError('P gains must be nonnegative')
        if self.localization_enabled and self.autostart:
            # 实机禁止未经人工授权即开始行走。仿真保留原 autostart 行为。
            raise ValueError('autostart must be false when localization_enabled=true')

        raw = p('~waypoints', [])
        if not isinstance(raw, list) or not raw:
            raise ValueError('~waypoints must be a nonempty YAML list')
        self.waypoints = []
        for i, entry in enumerate(raw):
            try:
                point = dict(id=str(entry.get('id', 'wp{}'.format(i + 1))),
                             x=float(entry['x']), y=float(entry['y']), yaw=float(entry['yaw']))
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise ValueError('Invalid waypoint {}: {}'.format(i, exc))
            if not all(math.isfinite(point[k]) for k in ('x', 'y', 'yaw')):
                raise ValueError('Non-finite waypoint {}'.format(i))
            self.waypoints.append(point)
        if any('wait' in point for point in raw):
            rospy.logwarn('Legacy waypoint wait fields ignored; manual next_waypoint required')

        # RLock: Subscriber、Timer、Service 与单调时钟看门狗并发访问状态。
        self.lock = threading.RLock()
        self.pose = None  # (x,y,yaw) in self.frame_id
        self.pose_wall_time = None
        self.pose_stamp = None
        self.gicp_status = None
        self.gicp_status_wall_time = None
        self.localization_pose_status = None
        self.localization_pose_status_wall_time = None
        self.fault_latched = False
        self.fault_reason = ''
        self.enabled = self.autostart
        self.index = 0
        self.state = 'WAIT_FOR_POSE' if self.autostart else 'IDLE'
        self.paused_at_waypoint = False
        self.shutdown_requested = False
        self.obstacle_value = 'FAULT_NO_HEARTBEAT'
        self.obstacle_goal_seq = -1
        self.obstacle_wall_time = None
        self.heartbeat_seq = 0

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher('/my_ros1/navigation/state', String,
                                          queue_size=5, latch=True)
        self.goal_pub = rospy.Publisher('/my_ros1/navigation/active_waypoint',
                                        PoseStamped, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher('/my_ros1/navigation/waypoints_path',
                                        Path, queue_size=1, latch=True)
        self.obstacle_goal_pub = rospy.Publisher('/my_ros1/navigation/obstacle_target',
                                                 String, queue_size=1, latch=True)
        # 不锁存！新订阅者必须等到实际控制周期发出心跳。
        self.heartbeat_pub = rospy.Publisher(self.heartbeat_topic, String,
                                              queue_size=1, latch=False)

        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(self.obstacle_topic, String, self.on_obstacle, queue_size=5)
        if self.localization_enabled:
            rospy.Subscriber(self.gicp_status_topic, String, self.on_gicp_status, queue_size=5)
            rospy.Subscriber(self.pose_status_topic, String, self.on_pose_status, queue_size=5)
        else:
            rospy.logwarn('LOCALIZATION GATE DISABLED (intended for simulation only)')

        rospy.Service('/my_ros1/navigation/start', Trigger, self.on_start)
        rospy.Service('/my_ros1/navigation/pause', Trigger, self.on_pause)
        rospy.Service('/my_ros1/navigation/next_waypoint', Trigger, self.on_next_waypoint)
        rospy.on_shutdown(self.on_shutdown)

        self.publish_path()
        self.publish_goal()
        self.publish_status()
        rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self.control_step)
        # 安全检查用 monotonic 线程，不依赖仿真 /clock：
        # 即便 rospy.Timer 随 /clock 停住，仍可尝试发送零速度。
        self.safety_thread = threading.Thread(target=self.safety_watchdog, daemon=True)
        self.safety_thread.start()
        rospy.loginfo('T800 controller: frame=%s, pose=%s, localization_enabled=%s, '
                      'waypoints=%d, heartbeat=%s',
                      self.frame_id, self.pose_topic, self.localization_enabled,
                      len(self.waypoints), self.heartbeat_topic)

    def on_pose(self, msg):
        """严格核对坐标系/时间戳；不拿陈旧消息刷新定位心跳。"""
        if msg.header.frame_id != self.frame_id:
            rospy.logwarn_throttle(5.0, 'Reject pose: frame %r expected %r',
                                   msg.header.frame_id, self.frame_id)
            return
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            rospy.logwarn_throttle(5.0, 'Reject pose with nonfinite position/quaternion')
            return
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-8:
            rospy.logwarn_throttle(5.0, 'Reject pose with zero quaternion')
            return
        stamp = msg.header.stamp
        if stamp.to_sec() <= 0:
            rospy.logwarn_throttle(5.0, 'Reject pose with zero timestamp')
            return
        now_ros = rospy.Time.now()
        if now_ros.to_sec() <= 0:
            rospy.logwarn_throttle(5.0, 'ROS clock not ready: reject pose')
            return
        age = (now_ros - stamp).to_sec()
        if age > self.pose_timeout or age < -self.pose_future_tolerance:
            rospy.logwarn_throttle(5.0, 'Reject stale/future pose (age %.3fs)', age)
            return
        yaw = euler_from_quaternion((q.x/norm, q.y/norm, q.z/norm, q.w/norm))[2]
        if not math.isfinite(yaw):
            return
        with self.lock:
            # 重复/回退的数据不得在收到时刻重置 pose_wall_time。
            if self.pose_stamp is not None and stamp <= self.pose_stamp:
                return
            self.pose = (p.x, p.y, yaw)
            self.pose_stamp = stamp
            self.pose_wall_time = time.monotonic()

    def pose_is_fresh(self):
        """锁内调用；同时检查接收时间与消息源时间。"""
        with self.lock:
            if self.pose is None or self.pose_wall_time is None or self.pose_stamp is None:
                return False
            if time.monotonic() - self.pose_wall_time > self.pose_timeout:
                return False
            now_ros = rospy.Time.now()
            if now_ros.to_sec() <= 0:
                return False
            age = (now_ros - self.pose_stamp).to_sec()
            return -self.pose_future_tolerance <= age <= self.pose_timeout

    def on_gicp_status(self, msg):
        with self.lock:
            self.gicp_status = msg.data.strip()
            self.gicp_status_wall_time = time.monotonic()
            # 一旦运行中重定位变为其他状态，立即锁定，不等待下一控制循环。
            if self.enabled and self.gicp_status != 'ALIGNED_INITIAL_ONLY':
                self.lock_localization_fault('GICP_STATUS_CHANGED:' + self.gicp_status)

    def on_pose_status(self, msg):
        with self.lock:
            self.localization_pose_status = msg.data.strip()
            self.localization_pose_status_wall_time = time.monotonic()
            if self.enabled and self.localization_pose_status != 'TRACKING_FIXED_GICP':
                self.lock_localization_fault('POSE_STATUS:' + self.localization_pose_status)

    def localization_health(self):
        """返回 (ready, reason)，必须在 RLock 内调用。"""
        with self.lock:
            if not self.pose_is_fresh():
                return False, 'POSE_MISSING_OR_STALE'
            if not self.localization_enabled:
                return True, 'SIM_DISABLED'
            now = time.monotonic()
            if (self.gicp_status_wall_time is None or
                    now - self.gicp_status_wall_time > self.gicp_status_timeout):
                return False, 'GICP_STATUS_TIMEOUT'
            if self.gicp_status != 'ALIGNED_INITIAL_ONLY':
                return False, 'GICP:' + str(self.gicp_status)
            if (self.localization_pose_status_wall_time is None or
                    now - self.localization_pose_status_wall_time > self.pose_status_timeout):
                return False, 'POSE_STATUS_TIMEOUT'
            if self.localization_pose_status != 'TRACKING_FIXED_GICP':
                return False, 'POSE_STATUS:' + str(self.localization_pose_status)
            return True, 'READY'

    def lock_localization_fault(self, reason):
        """一旦故障，停车并撤销使能；仅人工 /start 能解除。调用者须持锁。"""
        if not self.fault_latched:
            rospy.logerr('Localization safety lock: %s; manual /start required', reason)
        if self.state == 'WAIT_FOR_NEXT':
            self.paused_at_waypoint = True
        self.fault_latched = True
        self.fault_reason = reason
        self.enabled = False
        self.change_state('LOCALIZATION_FAULT_LOCKED')
        self.publish_cmd(0.0, 0.0)

    def on_obstacle(self, msg):
        fields = msg.data.strip().split('|')
        state = fields[0] if fields else 'FAULT_BAD_STATUS'
        seq = -1
        try:
            for field in fields[1:]:
                if field.startswith('goal='):
                    seq = int(field.split('=', 1)[1])
                    break
        except (TypeError, ValueError):
            state, seq = 'FAULT_BAD_STATUS', -1
        if state not in ('CLEAR', 'BLOCKED') and not state.startswith('FAULT_'):
            state = 'FAULT_BAD_STATUS'
        with self.lock:
            self.obstacle_value = state
            self.obstacle_goal_seq = seq
            self.obstacle_wall_time = time.monotonic()

    def obstacle_health(self):
        """仅当前目标的新鲜 CLEAR 允许行动。调用者须持锁。"""
        if not self.obstacle_enabled:
            return 'CLEAR'
        if (self.obstacle_wall_time is None or
                time.monotonic() - self.obstacle_wall_time > self.obstacle_timeout):
            return 'FAULT_COMM_TIMEOUT'
        if self.obstacle_goal_seq != self.index + 1:
            return 'FAULT_GOAL_MISMATCH'
        return self.obstacle_value

    def safety_watchdog(self):
        """独立真实时间安全线程；只做故障检查和停车，不生成存活心跳。"""
        interval = 1.0 / self.safety_poll_rate
        while not rospy.is_shutdown() and not self.shutdown_requested:
            with self.lock:
                if self.enabled:
                    ok, reason = self.localization_health()
                    if not ok:
                        self.lock_localization_fault(reason)
                    elif self.state not in ('WAIT_FOR_NEXT', 'FINISHED', 'IDLE'):
                        obs = self.obstacle_health()
                        if obs != 'CLEAR':
                            self.change_state('STOP_FOR_OBSTACLE' if obs == 'BLOCKED'
                                              else 'WAIT_FOR_OBSTACLE_DATA')
                            self.publish_cmd(0.0, 0.0)
            time.sleep(interval)

    def on_start(self, _req):
        """人工授权：数据恢复是必要条件，绝不自动解除定位故障锁。"""
        with self.lock:
            if self.shutdown_requested:
                return TriggerResponse(success=False, message='Controller shutting down')
            if self.enabled:
                if self.state == 'WAIT_FOR_NEXT':
                    return TriggerResponse(success=False,
                                           message='Call next_waypoint, not start')
                return TriggerResponse(success=False, message='Navigation already running')
            ok, reason = self.localization_health()
            if not ok:
                return TriggerResponse(success=False,
                                       message='Localization not ready: ' + reason)
            obs = self.obstacle_health()
            if obs != 'CLEAR':
                return TriggerResponse(success=False,
                                       message='Obstacle detector not CLEAR: ' + obs)
            # 只在所有先决条件全部就绪后解除故障锁。
            was_locked = self.fault_latched
            self.fault_latched = False
            self.fault_reason = ''
            if self.state == 'FINISHED' or self.index >= len(self.waypoints):
                self.index = 0
                self.paused_at_waypoint = False
            self.enabled = True
            if self.paused_at_waypoint:
                self.paused_at_waypoint = False
                self.change_state('WAIT_FOR_NEXT')
                message = 'Reauthorized, still waiting at waypoint {}'.format(self.index + 1)
            else:
                self.change_state('ALIGN_HEADING')
                message = ('Reauthorized' if was_locked else 'Started') + \
                          ' at waypoint {}'.format(self.index + 1)
            self.publish_goal()
            self.publish_cmd(0.0, 0.0)
            return TriggerResponse(success=True, message=message)

    def on_pause(self, _req):
        with self.lock:
            if self.state == 'WAIT_FOR_NEXT':
                self.paused_at_waypoint = True
            self.enabled = False
            # pause 不得清除故障锁、也不应覆盖故障状态。
            if not self.fault_latched and self.state != 'FINISHED':
                self.change_state('IDLE')
            self.publish_cmd(0.0, 0.0)
            return TriggerResponse(success=True,
                                   message='Paused; fault lock preserved if present')

    def on_next_waypoint(self, _req):
        with self.lock:
            if self.shutdown_requested or self.fault_latched or not self.enabled:
                return TriggerResponse(success=False,
                                       message='Not authorized; call start after resolving faults')
            if self.state != 'WAIT_FOR_NEXT':
                return TriggerResponse(success=False,
                                       message='Not waiting at waypoint: ' + self.state)
            ok, reason = self.localization_health()
            if not ok:
                self.lock_localization_fault(reason)
                return TriggerResponse(success=False, message='Localization fault: ' + reason)
            if self.index >= len(self.waypoints) - 1:
                return TriggerResponse(success=False, message='No next waypoint')
            self.index += 1
            self.obstacle_wall_time = None
            self.obstacle_goal_seq = -1
            self.publish_goal()
            self.change_state('ALIGN_HEADING')
            self.publish_cmd(0.0, 0.0)
            return TriggerResponse(success=True,
                                   message='Proceeding to waypoint {}/{} ({})'.format(
                                       self.index + 1, len(self.waypoints),
                                       self.waypoints[self.index]['id']))

    def change_state(self, state):
        if self.state != state:
            self.state = state
            self.publish_status()
            rospy.loginfo('Nav state => %s; waypoint %d/%d',
                          state, min(self.index + 1, len(self.waypoints)),
                          len(self.waypoints))

    def publish_status(self):
        label = self.waypoints[min(self.index, len(self.waypoints) - 1)]['id']
        self.status_pub.publish(String(
            data='state={} waypoint={}/{} id={} obstacle={} localization={} fault={}'.format(
                self.state, min(self.index + 1, len(self.waypoints)),
                len(self.waypoints), label, self.obstacle_health(),
                self.localization_pose_status if self.localization_enabled else 'SIM_DISABLED',
                self.fault_reason if self.fault_latched else 'NONE')))

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
            wp = self.waypoints[self.index]
            self.goal_pub.publish(self.as_pose(wp))
            self.obstacle_goal_pub.publish(String(
                data='{}|{:.9f}|{:.9f}'.format(self.index + 1, wp['x'], wp['y'])))

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        msg.poses = [self.as_pose(wp) for wp in self.waypoints]
        self.path_pub.publish(msg)

    def publish_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def turn_cmd(self, err):
        return saturated(signed_min_magnitude(
            saturated(self.kp_turn * err, self.max_turn), self.min_turn), self.max_turn)

    def publish_heartbeat(self):
        """仅 control_step 调用；控制回调卡死时心跳也会停止。"""
        self.heartbeat_seq += 1
        motion_states = ('ALIGN_HEADING', 'MOVE_FORWARD', 'ALIGN_FINAL_YAW')
        ready, reason = self.localization_health()
        obs = self.obstacle_health()
        motion_allowed = int(self.enabled and not self.fault_latched and
                             self.state in motion_states and ready and obs == 'CLEAR')
        # seq 不是安全时钟；ROS2 必须用本地 monotonic 测接收年龄。
        msg = ('seq={}|state={}|enabled={}|motion_allowed={}|'
               'localization={}|obstacle={}'.format(
                   self.heartbeat_seq, self.state, int(self.enabled), motion_allowed,
                   reason, obs))
        self.heartbeat_pub.publish(String(data=msg))

    def control_step(self, _event):
        with self.lock:
            # 重要：不在独立安全线程生成心跳；它反映控制回调存活。
            # 非活跃状态也发布心跳，但 motion_allowed 必须为 0。
            if self.shutdown_requested or self.fault_latched or not self.enabled:
                self.publish_cmd(0.0, 0.0)
                self.publish_heartbeat()
                return
            ok, reason = self.localization_health()
            if not ok:
                self.lock_localization_fault(reason)
                self.publish_heartbeat()
                return
            if self.state == 'WAIT_FOR_NEXT':
                self.publish_cmd(0.0, 0.0)
                self.publish_heartbeat()
                return
            obs = self.obstacle_health()
            if obs != 'CLEAR':
                self.change_state('STOP_FOR_OBSTACLE' if obs == 'BLOCKED'
                                  else 'WAIT_FOR_OBSTACLE_DATA')
                self.publish_cmd(0.0, 0.0)
                self.publish_heartbeat()
                return
            if self.state in ('WAIT_FOR_POSE', 'STOP_FOR_OBSTACLE',
                              'WAIT_FOR_OBSTACLE_DATA'):
                self.change_state('ALIGN_HEADING')

            x, y, yaw = self.pose
            goal = self.waypoints[self.index]
            dx, dy = goal['x'] - x, goal['y'] - y
            dist = math.hypot(dx, dy)
            if self.state == 'ALIGN_HEADING':
                if dist <= self.xy_tol:
                    self.change_state('ALIGN_FINAL_YAW')
                    self.publish_cmd(0.0, 0.0)
                else:
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
                else:
                    err = wrap_pi(math.atan2(dy, dx) - yaw)
                    if abs(err) >= self.realign:
                        self.change_state('ALIGN_HEADING')
                        self.publish_cmd(0.0, 0.0)
                    else:
                        vx = min(self.max_forward,
                                 max(self.min_forward, self.kp_forward * dist))
                        wz = saturated(self.kp_steer * err, self.max_steer)
                        self.publish_cmd(vx, wz)
            elif self.state == 'ALIGN_FINAL_YAW':
                err = wrap_pi(goal['yaw'] - yaw)
                if abs(err) <= self.yaw_tol:
                    self.publish_cmd(0.0, 0.0)
                    if self.index == len(self.waypoints) - 1:
                        self.enabled = False
                        self.change_state('FINISHED')
                    else:
                        self.change_state('WAIT_FOR_NEXT')
                else:
                    self.publish_cmd(0.0, self.turn_cmd(err))
            else:
                rospy.logwarn_throttle(5.0, 'Unknown navigation state %r', self.state)
                self.publish_cmd(0.0, 0.0)
            self.publish_heartbeat()

    def on_shutdown(self):
        with self.lock:
            self.shutdown_requested = True
            self.enabled = False
        # best effort；真正停止需下游 ROS2 超时看门狗及独立硬件急停。
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
