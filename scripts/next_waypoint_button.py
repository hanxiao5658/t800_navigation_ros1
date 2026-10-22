#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模拟展厅操作员点击平板上的“前往下一个位置”（ROS1 Noetic）。

使用方式（另开终端，A2sim ROS1）：
    rosrun t800_navigation next_waypoint_button.py

按 Enter：调用 /my_ros1/navigation/next_waypoint（std_srvs/Trigger）一次。
输入 q 再按 Enter：退出这个模拟按钮程序，不影响导航控制器。

只有 waypoint_controller 进入 WAIT_FOR_NEXT 状态后，按钮才能成功。
连续点击不会跳过中途 waypoint：行走中的调用会被拒绝。
"""

import rospy
from std_srvs.srv import Trigger


SERVICE = '/my_ros1/navigation/next_waypoint'


def main():
    rospy.init_node('next_waypoint_button_simulator', anonymous=True)
    rospy.loginfo('Waiting for %s ...', SERVICE)
    rospy.wait_for_service(SERVICE)
    next_waypoint = rospy.ServiceProxy(SERVICE, Trigger)

    print('\n模拟平板按钮：Enter = 前往下一个位置；q + Enter = 退出')
    while not rospy.is_shutdown():
        try:
            answer = input('点击 [前往下一个位置] > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\n按钮模拟器退出。')
            return
        if answer in ('q', 'quit', 'exit'):
            print('按钮模拟器退出。')
            return
        if answer:
            print('输入 Enter 表示点击；输入 q 退出。')
            continue
        try:
            response = next_waypoint()
            if response.success:
                print('[成功] {}'.format(response.message))
            else:
                print('[未执行] {}'.format(response.message))
        except rospy.ServiceException as exc:
            rospy.logerr('Calling %s failed: %s', SERVICE, exc)


if __name__ == '__main__':
    main()
