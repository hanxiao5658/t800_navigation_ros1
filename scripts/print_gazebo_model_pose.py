#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the XY position and yaw of one Gazebo model from /gazebo/model_states.

Usage (ROS1 Noetic):
  rosrun t800_navigation print_gazebo_model_pose.py _model_name:=unit_box

The model is selected by name, never by a hard-coded array index.
Yaw is reported in radians (for waypoints.yaml) and degrees (for inspection).
"""

import math
import time

import rospy
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


class GazeboModelPosePrinter:
    def __init__(self):
        self.model_name = rospy.get_param('~model_name', 'unit_box')
        self.print_rate = float(rospy.get_param('~print_rate', 1.0))
        if self.print_rate <= 0:
            raise ValueError('~print_rate must be greater than 0')
        self.last_print = -1.0
        self.last_warning = -1.0
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self.on_model_states, queue_size=1)
        rospy.loginfo('Reading model %r from /gazebo/model_states', self.model_name)

    def on_model_states(self, msg):
        now = time.monotonic()
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            if self.last_warning < 0 or now - self.last_warning >= 5.0:
                rospy.logwarn('Model %r not found. Available models: %s',
                              self.model_name, ', '.join(msg.name))
                self.last_warning = now
            return

        if self.last_print >= 0 and now - self.last_print < 1.0 / self.print_rate:
            return
        self.last_print = now

        pose = msg.pose[idx]
        q = pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        rospy.loginfo('%s | x=%.3f m, y=%.3f m, yaw=%.4f rad (%.1f deg)',
                      self.model_name,
                      pose.position.x, pose.position.y,
                      yaw, math.degrees(yaw))


def main():
    rospy.init_node('print_gazebo_model_pose', anonymous=True)
    GazeboModelPosePrinter()
    rospy.spin()


if __name__ == '__main__':
    main()
