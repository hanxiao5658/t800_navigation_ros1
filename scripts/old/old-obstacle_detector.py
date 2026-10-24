#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 展厅前方通道障碍物检测（ROS1 / Gazebo 第一版）。

目的：
  1. 订阅原始 MID360 CustomMsg（无需 FAST-LIO）；
  2. 通过 Gazebo /gazebo/link_states 的 dummy_root / mid360_link 真值姿态，
     正确考虑雷达安装的 20° pitch 以及胸部关节变化；
  3. 在机器人机体前方 [min_forward, max_forward] × corridor_width 的走廊内找障碍；
  4. 同时沿"当前位置 -> 当前 Waypoint"方向裁剪检测范围，避免目标点后方墙壁误触发；
  5. 持续发布 CLEAR / BLOCKED / FAULT_* 状态与真实时间心跳。

话题：
  订阅  /livox/lidar                          livox_laser_simulation/CustomMsg
        /gazebo/link_states                  gazebo_msgs/LinkStates
        /my_ros1/navigation/base_pose_gt    geometry_msgs/PoseStamped
        /my_ros1/navigation/obstacle_target  std_msgs/String ("序号|x|y")
  发布  /my_ros1/navigation/obstacle_status std_msgs/String

状态数据格式："CLEAR|goal=2|points=0|nearest=nan"
控制器只有收到当前 goal 序号对应的、新鲜的 CLEAR 才允许运动。

边界：
  - 此版本使用 Gazebo 真值坐标变换，不是可直接部署实机的驱动；实机需替换为
    TF + 里程计，并与雷达时间同步。
  - 源激光仿真存在机器人自身回波。下面的 self-filter 是保守几何近似，
    会牺牲贴身小范围的检测能力。真正实机请使用自模型或碰撞体自过滤。
  - 当前不是安全认证传感器；不能代替实机急停、制动试验。
"""

import math
import threading
import time

import numpy as np
import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped
from livox_laser_simulation.msg import CustomMsg
from std_msgs.msg import String
from tf.transformations import quaternion_matrix


def transform_of(pose):
    """geometry_msgs/Pose -> 4x4 齐次矩阵，旋转和平移全部保留。"""
    q = pose.orientation
    t = quaternion_matrix((q.x, q.y, q.z, q.w))
    t[:3, 3] = (pose.position.x, pose.position.y, pose.position.z)
    return t


class ObstacleDetector:
    def __init__(self):
        rospy.init_node('obstacle_detector')

        # 以下均为 ROS 私有参数，由 obstacle_params.yaml 加载。
        gp = rospy.get_param
        self.lidar_topic = gp('~lidar_topic', '/livox/lidar')
        self.links_topic = gp('~links_topic', '/gazebo/link_states')
        self.pose_topic = gp('~pose_topic', '/my_ros1/navigation/base_pose_gt')
        self.goal_topic = gp('~goal_topic', '/my_ros1/navigation/obstacle_target')
        self.status_topic = gp('~status_topic', '/my_ros1/navigation/obstacle_status')
        self.base_link_name = gp('~base_link_name', 't800::dummy_root')
        self.lidar_link_name = gp('~lidar_link_name', 't800::mid360_link')

        # 主检测区域为机体 +X 方向，宽度围绕机体左右对称。
        self.min_forward = float(gp('~min_forward', 0.2))
        self.max_forward = float(gp('~max_forward', 2.0))
        self.half_width = float(gp('~corridor_half_width', 0.5))
        # Gazebo 世界坐标 Z 裁剪地面与天花板；展厅地面约 z=0。
        self.min_world_z = float(gp('~min_world_z', 0.12))
        self.max_world_z = float(gp('~max_world_z', 1.85))
        # 不检测 Waypoint 后面的墙：沿路线投影不得超过距离目标的距离 - margin。
        self.goal_end_margin = float(gp('~goal_end_margin', 0.03))
        self.goal_close_distance = float(gp('~goal_close_distance', 0.05))

        # 排除上半身自身回波的近似遮罩；不适合替代精确几何 self-filter。
        self.self_filter_enabled = bool(gp('~self_filter_enabled', True))
        self.self_max_x = float(gp('~self_max_x', 0.4))
        self.self_half_width = float(gp('~self_half_width', 0.48))
        self.self_min_world_z = float(gp('~self_min_world_z', 0.75))

        self.min_obstacle_points = int(gp('~min_obstacle_points', 5))
        self.min_valid_points = int(gp('~min_valid_points', 15))
        self.block_confirm_scans = int(gp('~block_confirm_scans', 2))
        self.clear_confirm_scans = int(gp('~clear_confirm_scans', 3))
        self.cloud_timeout = float(gp('~cloud_timeout', 0.5))
        self.pose_timeout = float(gp('~pose_timeout', 0.5))
        self.links_timeout = float(gp('~links_timeout', 0.5))
        self.goal_timeout = float(gp('~goal_timeout', 0.0))  # 0：目标 latched，永不过期
        self.status_rate = float(gp('~status_rate', 10.0))

        if not (0 <= self.min_forward < self.max_forward and
                self.min_world_z < self.max_world_z and self.half_width > 0 and
                self.goal_end_margin >= 0 and self.min_obstacle_points >= 1 and
                self.min_valid_points >= 1 and self.block_confirm_scans >= 1 and
                self.clear_confirm_scans >= 1 and self.cloud_timeout > 0 and
                self.pose_timeout > 0 and self.links_timeout > 0 and self.status_rate > 0):
            raise ValueError('Invalid obstacle detector parameters')

        self.lock = threading.RLock()
        self.links = None  # (world_T_base, world_T_lidar)
        self.links_at = None
        self.base_pose = None  # ROS1 标准 PoseStamped，与控制器使用同一世界坐标系
        self.pose_at = None
        self.goal = None  # (seq, x, y)
        self.goal_at = None
        self.cloud_at = None  # 收到且成功评估的最新一次点云的本地 monotonic 时刻
        self.scan_goal_seq = None
        self.scan_result = 'FAULT_WAIT_SCAN'
        self.blocked = False
        self.clear_confirmed = False
        self.block_streak = 0
        self.clear_streak = 0
        self.count = 0
        self.nearest = math.nan

        self.pub = rospy.Publisher(self.status_topic, String, queue_size=10)
        rospy.Subscriber(self.links_topic, LinkStates, self.on_links, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(self.goal_topic, String, self.on_goal, queue_size=1)
        rospy.Subscriber(self.lidar_topic, CustomMsg, self.on_cloud,
                         queue_size=1, buff_size=8 * 1024 * 1024)

        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.status_loop, daemon=True)
        self.thread.start()
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo('Obstacle detector: %.2f..%.2f m, width=%.2f m, '
                      'goal end margin=%.2f m, sensor=%s',
                      self.min_forward, self.max_forward, 2*self.half_width,
                      self.goal_end_margin, self.lidar_topic)

    def on_links(self, msg):
        try:
            ib = msg.name.index(self.base_link_name)
            il = msg.name.index(self.lidar_link_name)
        except ValueError:
            rospy.logwarn_throttle(5.0, 'Missing Gazebo link %s or %s',
                                   self.base_link_name, self.lidar_link_name)
            return
        try:
            world_base = transform_of(msg.pose[ib])
            world_lidar = transform_of(msg.pose[il])
        except Exception as exc:
            rospy.logwarn_throttle(5.0, 'Invalid Gazebo link pose: %s', str(exc))
            return
        with self.lock:
            self.links = (world_base, world_lidar)
            self.links_at = time.monotonic()

    def on_pose(self, msg):
        with self.lock:
            self.base_pose = (float(msg.pose.position.x), float(msg.pose.position.y))
            self.pose_at = time.monotonic()

    def on_goal(self, msg):
        # 原子消息 '序号|x|y'；不依赖 PoseStamped.header.seq 自动编号行为。
        try:
            parts = msg.data.split('|')
            if len(parts) != 3:
                raise ValueError('Expected goal format: seq|x|y')
            goal = (int(parts[0]), float(parts[1]), float(parts[2]))
            if goal[0] < 1 or not all(math.isfinite(v) for v in goal[1:]):
                raise ValueError('Invalid goal index or XY')
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(5.0, 'Bad obstacle_target: %s', str(exc))
            return
        with self.lock:
            if self.goal != goal:
                # 新目标：清空旧目标上累计的状态；首帧点云算完之前不能报告 CLEAR。
                self.block_streak = self.clear_streak = 0
                self.blocked = False
                self.clear_confirmed = False
                self.scan_goal_seq = None
                self.scan_result = 'FAULT_WAIT_SCAN'
            self.goal = goal
            self.goal_at = time.monotonic()

    def on_cloud(self, msg):
        """每收到一帧点云评估一次，滤波判定次数以真实新扫描帧计算。"""
        now = time.monotonic()
        with self.lock:
            links = self.links
            pose = self.base_pose
            goal = self.goal
            links_at, pose_at = self.links_at, self.pose_at
        if links is None or pose is None or goal is None:
            return
        if (now - links_at > self.links_timeout or now - pose_at > self.pose_timeout):
            return

        # 解析 Gazebo Livox CustomMsg，过滤无效的全零点、Inf 和 NaN。
        # 10k points @10Hz: NumPy 批量计算比纯 Python for 循环低开销。
        pts = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float64)
        if pts.size == 0:
            with self.lock:
                self.cloud_at = now
                self.scan_goal_seq = goal[0]
                self.scan_result = 'FAULT_SPARSE_CLOUD'
            return
        mask = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.05)
        pts = pts[mask]
        if len(pts) < self.min_valid_points:
            with self.lock:
                self.cloud_at = now
                self.scan_goal_seq = goal[0]
                self.scan_result = 'FAULT_SPARSE_CLOUD'
            return

        world_base, world_lidar = links
        # R_world_lidar @ 点 + t_world_lidar，注意这一步自动包含20°安装俯仰。
        world = pts @ world_lidar[:3, :3].T + world_lidar[:3, 3]
        # world -> base：用机器人底座真实三维姿态，剔除动态俯仰影响。
        body = (world - world_base[:3, 3]) @ world_base[:3, :3]

        seq, gx, gy = goal
        bx, by = pose
        dx, dy = gx - bx, gy - by
        distance = math.hypot(dx, dy)

        if distance <= self.goal_close_distance:
            obstacle_count, nearest = 0, math.nan
        else:
            # 动态裁剪终点：只检查行走到当前 waypoint 之前的路径，
            # 后方墙体沿路径投影大于 distance，必定被排除。
            max_path_s = min(self.max_forward, max(0.0, distance-self.goal_end_margin))
            ux, uy = dx/distance, dy/distance
            along = (world[:, 0] - bx)*ux + (world[:, 1] - by)*uy
            cross = -(world[:, 0] - bx)*uy + (world[:, 1] - by)*ux

            relevant = (
                (body[:, 0] >= self.min_forward) &
                (body[:, 0] <= self.max_forward) &
                (np.abs(body[:, 1]) <= self.half_width) &
                (world[:, 2] >= self.min_world_z) &
                (world[:, 2] <= self.max_world_z) &
                (along >= self.min_forward) &
                (along <= max_path_s) &
                (np.abs(cross) <= self.half_width)
            )
            if self.self_filter_enabled:
                # 简单近身、自体的上半身回波遮罩。真实近距离盲区须单独评估。
                self_points = ((body[:, 0] <= self.self_max_x) &
                               (np.abs(body[:, 1]) <= self.self_half_width) &
                               (world[:, 2] >= self.self_min_world_z))
                relevant &= ~self_points
            obstacle_count = int(relevant.sum())
            nearest = float(body[relevant, 0].min()) if obstacle_count else math.nan

        with self.lock:
            # 如果处理点云期间用户切换了目标，禁止把旧目标状态写回当前目标。
            if self.goal != goal:
                return
            self.cloud_at = now
            self.scan_goal_seq = seq
            self.count = obstacle_count
            self.nearest = nearest
            if obstacle_count >= self.min_obstacle_points:
                self.block_streak += 1
                self.clear_streak = 0
                if self.block_streak >= self.block_confirm_scans:
                    self.blocked = True
                # 未达到阻挡确认帧数之前，不能当作安全无障碍。
                self.scan_result = ('BLOCKED' if self.blocked
                                    else 'FAULT_CONFIRMING_BLOCK')
            else:
                self.clear_streak += 1
                self.block_streak = 0
                if self.clear_streak >= self.clear_confirm_scans:
                    self.blocked = False
                    self.clear_confirmed = True
                self.scan_result = ('BLOCKED' if self.blocked else
                                    'CLEAR' if self.clear_confirmed else
                                    'FAULT_CONFIRMING_CLEAR')

    def status_payload(self):
        """独立于 Gazebo /clock 的健康判定：任何输入缺失均返回 FAULT。"""
        now = time.monotonic()
        with self.lock:
            seq = self.goal[0] if self.goal else 0
            count, nearest = self.count, self.nearest
            if self.goal is None:
                status = 'FAULT_NO_GOAL'
            elif self.links is None or now - self.links_at > self.links_timeout:
                status = 'FAULT_LINKS_TIMEOUT'
            elif self.base_pose is None or now - self.pose_at > self.pose_timeout:
                status = 'FAULT_POSE_TIMEOUT'
            elif self.cloud_at is None or now - self.cloud_at > self.cloud_timeout:
                status = 'FAULT_LIDAR_TIMEOUT'
            elif self.scan_goal_seq != seq:
                status = 'FAULT_WAIT_SCAN'
            else:
                status = self.scan_result
        return '{}|goal={}|points={}|nearest={:.3f}'.format(
            status, seq, count, nearest)

    def status_loop(self):
        # 不用 rospy.Timer：use_sim_time=true 的 Gazebo 暂停时，也能发故障心跳。
        period = 1.0/self.status_rate
        last_display = ''
        while not rospy.is_shutdown() and not self.stop_flag.is_set():
            value = self.status_payload()
            self.pub.publish(String(data=value))
            state = value.split('|', 1)[0]
            if state != last_display:
                rospy.loginfo('Obstacle detector => %s', value)
                last_display = state
            self.stop_flag.wait(period)

    def on_shutdown(self):
        self.stop_flag.set()


if __name__ == '__main__':
    try:
        ObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
