#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将定位仓库输出的 base_footprint_pose 转换为便于人工记录 Waypoint 的文本。

订阅：
  /my_ros1/localization/base_footprint_pose
  类型：geometry_msgs/PoseStamped

发布：
  /my_ros1/localization/base_footprint_pose_xyYaw
  类型：std_msgs/String

输出包括 frame_id、x、y、z、roll、pitch、yaw（角度单位均为 rad）。
yaw_deg 仅用于人工查看，不要将它填入 waypoint 的 yaw 字段。

本脚本不做坐标系变换，只保留输入位置并将四元数转换为 RPY。
只有输入 frame_id 为 map 时才发布，避免把其他坐标系的坐标误记为 map。
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


# ============================================================
# 可修改参数：通常不需要修改
# ============================================================

INPUT_TOPIC = "/my_ros1/localization/base_footprint_pose"
OUTPUT_TOPIC = "/my_ros1/localization/base_footprint_pose_xyYaw"

# 只接受 map 坐标系，防止记录错误坐标系下的 Waypoint
EXPECTED_FRAME_ID = "map"

# 输出小数位数
DECIMAL_PLACES = 6


class BaseFootprintPoseToXYYaw:
    def __init__(self):
        self.publisher = rospy.Publisher(
            OUTPUT_TOPIC, String, queue_size=10
        )
        self.subscriber = rospy.Subscriber(
            INPUT_TOPIC, PoseStamped, self.pose_callback, queue_size=10
        )

        rospy.loginfo("订阅位姿：%s", INPUT_TOPIC)
        rospy.loginfo("发布转换结果：%s", OUTPUT_TOPIC)
        rospy.loginfo("要求输入坐标系：%s", EXPECTED_FRAME_ID)

    def pose_callback(self, msg):
        # 1. 检查输入坐标系。不是 map 就不发布，避免记录错坐标。
        frame_id = msg.header.frame_id.lstrip("/")
        if frame_id != EXPECTED_FRAME_ID:
            rospy.logwarn_throttle(
                2.0,
                "收到的 frame_id 是 '%s'，不是 '%s'，本帧不发布。",
                msg.header.frame_id,
                EXPECTED_FRAME_ID
            )
            return

        # 2. 保留输入位置坐标 x、y、z
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # 3. 读取四元数。ROS 顺序为 x、y、z、w。
        q = msg.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]

        # 检查并归一化四元数，避免无效数据产生错误角度
        q_norm = math.sqrt(sum(value * value for value in quaternion))
        if q_norm < 1e-8:
            rospy.logwarn_throttle(
                2.0, "收到无效四元数（模长接近 0），本帧不发布。"
            )
            return

        quaternion = [value / q_norm for value in quaternion]

        # 转换得到 roll、pitch、yaw，单位均为 rad
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # 转成 degree 仅供人查看；waypoint 的 yaw 使用 rad
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        # 4. 输出为可读文本，方便 rostopic echo 后复制到 waypoint YAML
        # 请复制 x、y、yaw 三项；yaw 是 rad，yaw_deg 仅供查看。
        d = DECIMAL_PLACES
        output_text = (
            "frame_id: {frame}\n"
            "x: {x:.{d}f}\n"
            "y: {y:.{d}f}\n"
            "z: {z:.{d}f}\n"
            "roll: {roll:.{d}f}       # rad\n"
            "pitch: {pitch:.{d}f}     # rad\n"
            "yaw: {yaw:.{d}f}         # rad，复制到 waypoint 的 yaw\n"
            "roll_deg: {roll_deg:.3f}   # degree，仅供查看\n"
            "pitch_deg: {pitch_deg:.3f} # degree，仅供查看\n"
            "yaw_deg: {yaw_deg:.3f}     # degree，仅供查看"
        ).format(
            frame=frame_id,
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            d=d
        )

        self.publisher.publish(String(data=output_text))

        # 每 2 秒打印一次摘要，避免高频刷屏
        rospy.loginfo_throttle(
            2.0,
            "Map pose: x=%.3f, y=%.3f, yaw=%.4f rad (%.2f deg)",
            x, y, yaw, yaw_deg
        )


def main():
    rospy.init_node("base_footprint_pose_to_xyyaw")
    BaseFootprintPoseToXYYaw()

    rospy.loginfo("Waypoint 位姿转换节点已启动。按 Ctrl+C 退出。")
    rospy.spin()


if __name__ == "__main__":
    main()
