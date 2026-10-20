#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish the Gazebo T800 base-link ground truth as ROS1 PoseStamped.

Input:  /gazebo/link_states (gazebo_msgs/LinkStates)
Output: /my_ros1/navigation/base_pose_gt (geometry_msgs/PoseStamped)

The dummy_root link is fixed to LINK_BASE in the visual-only T800 model.
"""

import threading
import time

import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped


class GazeboGroundTruth:
    def __init__(self):
        rospy.init_node('gazebo_ground_truth')
        self.link_name = rospy.get_param('~link_name', 't800::dummy_root')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.input_topic = rospy.get_param('~input_topic', '/gazebo/link_states')
        self.output_topic = rospy.get_param(
            '~output_topic', '/my_ros1/navigation/base_pose_gt')
        rate = float(rospy.get_param('~publish_rate', 30.0))
        if rate <= 0:
            raise ValueError('publish_rate must be positive')
        self.timeout = float(rospy.get_param('~source_timeout', 0.5))

        self.lock = threading.Lock()
        self.latest_pose = None
        self.latest_wall_time = None
        self.pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=5)
        rospy.Subscriber(self.input_topic, LinkStates, self.on_links, queue_size=1)
        rospy.Timer(rospy.Duration.from_sec(1.0 / rate), self.publish_pose)
        rospy.loginfo('Gazebo truth %s -> %s (link=%s)',
                      self.input_topic, self.output_topic, self.link_name)

    def on_links(self, msg):
        try:
            idx = msg.name.index(self.link_name)
        except ValueError:
            rospy.logwarn_throttle(5.0, 'Gazebo link not found: %s', self.link_name)
            return
        if idx >= len(msg.pose):
            return
        with self.lock:
            self.latest_pose = msg.pose[idx]
            self.latest_wall_time = time.monotonic()

    def publish_pose(self, _event):
        with self.lock:
            pose = self.latest_pose
            t = self.latest_wall_time
        if pose is None or t is None:
            return
        if self.timeout > 0 and time.monotonic() - t > self.timeout:
            rospy.logwarn_throttle(5.0, 'Gazebo truth data stale; withholding pose')
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()  # Gazebo /clock if /use_sim_time=true
        msg.header.frame_id = self.world_frame
        msg.pose = pose
        self.pub.publish(msg)


if __name__ == '__main__':
    try:
        GazeboGroundTruth()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
