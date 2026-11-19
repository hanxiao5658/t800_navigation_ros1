#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 ROS1 障碍物检测器独立测试：持续发布模拟机器人位姿和导航目标。

注意：
  1. 只发布两个测试话题，不向机器人发布运动指令。
  2. 不使用 launch/YAML/ROS 参数服务器来设置测试位姿；直接修改下方【可修改参数区】。
  3. 仅在导航控制器和真实定位节点没有使用这些测试话题进行实机控制时运行；
     尤其不要与真正的 base_footprint_pose 发布者混用，以免影响导航安全。
  4. obstacle_detector.py 的 obstacle_target 消息格式固定为 "seq|x|y"，
     所以目标 z/yaw 虽然保留为可编辑变量，但目前不会发到目标话题或参与避障判断。
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# ===========================================================================
# 【可修改参数区】—— 要改位置/朝向，只修改这里，不用修改其他代码
# ===========================================================================

# ① /my_ros1/localization/base_footprint_pose —— 机器人当前位置
#    POSE_X:   机器人在 map 坐标系的 X (米)
#    POSE_Y:   机器人在 map 坐标系的 Y (米)
#    POSE_Z:   机器人在 map 坐标系的 Z (米)
#    POSE_YAW_DEG: 机器人朝向 (角度)，0° 表示朝 map 的 +X 方向，
#                  +90° 表示朝 +Y 方向，-90° 表示朝 -Y 方向。
POSE_X = 0.0
POSE_Y = 0.0
POSE_Z = 0.0
POSE_YAW_DEG = 0.0

# ② /my_ros1/navigation/obstacle_target —— 当前导航目标
#    GOAL_X:   目标在 map 坐标系的 X (米)
#    GOAL_Y:   目标在 map 坐标系的 Y (米)
#    GOAL_Z:   保留供人工记录；现有检测器不读取该值
#    GOAL_YAW_DEG: 保留供人工记录；现有检测器不读取该值
#    GOAL_SEQ: 当前导航目标的序号（正整数），通常首个 waypoint 为 1；
#              当测试与 waypoint_controller 同时使用时，必须与其当前目标序号一致。
GOAL_X = 5.0
GOAL_Y = 0.0
GOAL_Z = 0.0
GOAL_YAW_DEG = 0.0
GOAL_SEQ = 1

# 发布频率：位姿与目标均持续发布，10Hz 足以满足检测器默认 0.35 秒位姿超时。
PUBLISH_RATE_HZ = 10.0

# 与 obstacle_params_real.yaml 中的 world_frame: map 保持一致。
POSE_FRAME_ID = "map"

# ===========================================================================
# 以下是发布实现，通常不需要修改
# ===========================================================================


def make_pose_message(stamp):
    """将 (x,y,z,yaw) 转成 geometry_msgs/PoseStamped。"""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = POSE_FRAME_ID
    msg.pose.position.x = POSE_X
    msg.pose.position.y = POSE_Y
    msg.pose.position.z = POSE_Z

    # 当前测试姿态只设置 Yaw；Roll/Pitch 固定为 0。
    yaw_rad = math.radians(POSE_YAW_DEG)
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
    msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
    return msg


def make_target_message():
    """匹配 obstacle_detector.py 的 on_goal(): 'seq|x|y'。"""
    return String(data="{}|{:.9f}|{:.9f}".format(GOAL_SEQ, GOAL_X, GOAL_Y))


def main():
    if GOAL_SEQ <= 0:
        raise ValueError("GOAL_SEQ 必须为正整数")
    if PUBLISH_RATE_HZ <= 0:
        raise ValueError("PUBLISH_RATE_HZ 必须大于 0")
    for name, value in (
        ("POSE_X", POSE_X), ("POSE_Y", POSE_Y), ("POSE_Z", POSE_Z),
        ("POSE_YAW_DEG", POSE_YAW_DEG), ("GOAL_X", GOAL_X),
        ("GOAL_Y", GOAL_Y), ("GOAL_Z", GOAL_Z),
        ("GOAL_YAW_DEG", GOAL_YAW_DEG),
    ):
        if not math.isfinite(value):
            raise ValueError("{} 必须是有限数值".format(name))

    rospy.init_node("test_obstacle_dector", anonymous=False)

    pose_pub = rospy.Publisher(
        "/my_ros1/localization/base_footprint_pose", PoseStamped, queue_size=10
    )
    target_pub = rospy.Publisher(
        "/my_ros1/navigation/obstacle_target", String, queue_size=10, latch=True
    )

    target_msg = make_target_message()
    rospy.loginfo(
        "[TEST] base_footprint_pose: map (x=%.3f, y=%.3f, z=%.3f, yaw=%.1f deg)",
        POSE_X, POSE_Y, POSE_Z, POSE_YAW_DEG,
    )
    rospy.loginfo("[TEST] obstacle_target: %s", target_msg.data)
    rospy.logwarn(
        "[TEST] GOAL_Z=%.3f, GOAL_YAW_DEG=%.1f are NOT used by obstacle_detector.py",
        GOAL_Z, GOAL_YAW_DEG,
    )
    rospy.logwarn("[TEST] 请确保真实导航运动未使能，且不要同时发布真实定位位姿。")

    rate = rospy.Rate(PUBLISH_RATE_HZ)
    while not rospy.is_shutdown():
        # 每次刷新时间戳，避免障碍检测器把位姿判断成过期数据。
        pose_pub.publish(make_pose_message(rospy.Time.now()))
        target_pub.publish(target_msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
