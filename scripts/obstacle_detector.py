#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 ROS1 前向障碍物检测：六自由度固定外参版（仿真/实机接口一致）。

坐标约定（务必先在机器人实机上标定）：
  foot_frame：双脚并拢、两脚脚尖连线中点；+X 前、+Y 左、+Z 上。
  lidar_xyz / lidar_rpy：MID360 光学/消息坐标系相对 foot_frame 的位姿。
  p_foot = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ p_lidar + t_lidar

订阅：
  /livox/lidar                            CustomMsg (类型由 lidar_message_type 指定)
  /my_ros1/navigation/base_pose_gt        geometry_msgs/PoseStamped
  /my_ros1/navigation/obstacle_target     std_msgs/String, "seq|x|y"
发布：
  /my_ros1/navigation/obstacle_status     std_msgs/String, 保持与控制器兼容
  /my_ros1/navigation/obstacle_debug      std_msgs/String, 滤波逐层诊断

不再订阅任何 /gazebo/* 链接真值。障碍检测使用 MID360 原始点云，
不依赖 FAST-LIO 点云或定位状态。当前位置和目标仍需位姿来源；
当前导航使用 Gazebo 真值，实机时更换 pose_topic 及参照点标定即可。

状态：CLEAR/BLOCKED/FAULT_*，控制器只允许当前目标新鲜 CLEAR 放行。
本节点和下游导航控制器都使用 monotonic 时间监控通信超时。
"""

import importlib
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


def rotation_from_rpy(roll, pitch, yaw):
    """ROS URDF 标准固定轴 RPY：R=Rz(yaw)Ry(pitch)Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                  cp * cr],
    ], dtype=np.float64)


def filter_points(points_foot, goal_in_foot, cfg):
    """纯几何核心（便于脱离 ROS 单测），返回障碍点数量及逐层诊断。

    points_foot: Nx3，固定外参变换到足部标定坐标系后的点云。
    goal_in_foot: (x,y)，当前导航目标相对足部原点的平面坐标。
    注意：只扫描 *目标之前* 的直线走廊，不把目标背后的墙体算作障碍物。
    """
    diag = dict(front=0, path=0, height=0, self_removed=0,
                nearest_front=math.nan, nearest_path=math.nan,
                nearest_height=math.nan, nearest=math.nan, goal_distance=0.0)
    if len(points_foot) == 0:
        return 0, diag

    x, y, z = points_foot[:, 0], points_foot[:, 1], points_foot[:, 2]
    gx, gy = goal_in_foot
    distance = math.hypot(gx, gy)
    diag['goal_distance'] = distance

    # 原始机体前方2m长方形；此计数不应用高度、自过滤或目标裁剪。
    front = ((x >= cfg['min_forward']) &
             (x <= cfg['max_forward']) &
             (np.abs(y) <= cfg['corridor_half_width']))
    diag['front'] = int(front.sum())
    if diag['front']:
        diag['nearest_front'] = float(x[front].min())

    # 剩余目标距离小于容差时，不再对目标身后的墙报警。
    # 注意：此机制不等于对原地转向或机器人完整身体的安全碰撞检查。
    if distance <= cfg['goal_close_distance']:
        return 0, diag

    ux, uy = gx / distance, gy / distance
    along = x * ux + y * uy
    cross = -x * uy + y * ux
    end_s = min(cfg['max_forward'], max(0.0, distance - cfg['goal_end_margin']))

    path = (front & (along >= cfg['min_forward']) &
            (along <= end_s) &
            (np.abs(cross) <= cfg['corridor_half_width']))
    diag['path'] = int(path.sum())
    if diag['path']:
        diag['nearest_path'] = float(x[path].min())

    # 世界Z已被废弃：Z 是相对足部地面的高度，与传感器安装俯仰独立。
    height = (path & (z >= cfg['min_obstacle_z']) &
              (z <= cfg['max_obstacle_z']))
    diag['height'] = int(height.sum())
    if diag['height']:
        diag['nearest_height'] = float(x[height].min())

    valid = height
    if cfg['self_filter_enabled']:
        # 第一版胸/手臂粗略几何遮罩：此区域可能同时排除真实障碍。
        self_points = ((x <= cfg['self_max_x']) &
                       (np.abs(y) <= cfg['self_half_width']) &
                       (z >= cfg['self_min_z']))
        diag['self_removed'] = int((height & self_points).sum())
        valid = height & (~self_points)

    n = int(valid.sum())
    if n:
        diag['nearest'] = float(x[valid].min())
    return n, diag


class ObstacleDetector:
    def __init__(self):
        rospy.init_node('obstacle_detector')
        p = rospy.get_param
        self.lidar_topic = p('~lidar_topic', '/livox/lidar')
        self.lidar_message_type = p('~lidar_message_type',
                                    'livox_laser_simulation/CustomMsg')
        try:
            msg_pkg, msg_name = self.lidar_message_type.split('/')
            lidar_cls = getattr(importlib.import_module(msg_pkg + '.msg'), msg_name)
        except (ValueError, ImportError, AttributeError) as exc:
            raise RuntimeError('Cannot import lidar_message_type {}: {}'.format(
                self.lidar_message_type, exc))
        self.pose_topic = p('~pose_topic', '/my_ros1/navigation/base_pose_gt')
        self.goal_topic = p('~goal_topic', '/my_ros1/navigation/obstacle_target')
        self.status_topic = p('~status_topic', '/my_ros1/navigation/obstacle_status')
        self.debug_topic = p('~debug_topic', '/my_ros1/navigation/obstacle_debug')
        self.expected_world_frame = p('~world_frame', 'world')

        # 注意：这不是 URDF 中相对 LINK_WAIST_YAW 的 xyz。
        # 标定参考是双脚脚尖连线中点（foot_frame），单位 m/rad。
        self.extrinsics_calibrated = bool(p('~extrinsics_calibrated', False))
        self.tx = float(p('~lidar_x', 0.0))
        self.ty = float(p('~lidar_y', 0.0))
        self.tz = float(p('~lidar_z', 1.2))
        roll = float(p('~lidar_roll', 0.0))
        pitch = float(p('~lidar_pitch', 0.8722))
        yaw = float(p('~lidar_yaw', 0.0))
        if not all(math.isfinite(a) for a in (self.tx, self.ty, self.tz,
                                               roll, pitch, yaw)):
            raise ValueError('Non-finite MID360 extrinsic')
        self.sensor_R = rotation_from_rpy(roll, pitch, yaw)
        self.sensor_t = np.array((self.tx, self.ty, self.tz), dtype=np.float64)

        # 若 pose_topic 中的 XY 不是脚尖中点，可设置以下平面偏移：
        # pose_origin_to_foot_x/y 指足部原点相对 pose_topic 位姿原点的平移，
        # 在 pose_topic 的 yaw 旋转坐标系下表达。
        # 当前 Gazebo 的 pose_topic 是 dummy_root，而非足部原点。
        self.pose_origin_to_foot_x = float(p('~pose_origin_to_foot_x', 0.0))
        self.pose_origin_to_foot_y = float(p('~pose_origin_to_foot_y', 0.0))

        self.cfg = dict(
            min_forward=float(p('~min_forward', 0.2)),
            max_forward=float(p('~max_forward', 2.0)),
            corridor_half_width=float(p('~corridor_half_width', 0.5)),
            min_obstacle_z=float(p('~min_obstacle_z', 0.12)),
            max_obstacle_z=float(p('~max_obstacle_z', 1.85)),
            goal_end_margin=float(p('~goal_end_margin', 0.03)),
            goal_close_distance=float(p('~goal_close_distance', 0.05)),
            self_filter_enabled=bool(p('~self_filter_enabled', True)),
            self_max_x=float(p('~self_max_x', 0.4)),
            self_half_width=float(p('~self_half_width', 0.48)),
            self_min_z=float(p('~self_min_z', 0.75)),
        )
        self.min_obstacle_points = int(p('~min_obstacle_points', 5))
        self.min_valid_points = int(p('~min_valid_points', 15))
        self.block_confirm_scans = int(p('~block_confirm_scans', 1))
        self.clear_confirm_scans = int(p('~clear_confirm_scans', 3))
        self.cloud_timeout = float(p('~cloud_timeout', 0.35))
        self.pose_timeout = float(p('~pose_timeout', 0.35))
        self.status_rate = float(p('~status_rate', 10.0))
        self.debug_rate = float(p('~debug_rate', 2.0))

        c = self.cfg
        if not (0 <= c['min_forward'] < c['max_forward'] and
                c['corridor_half_width'] > 0 and
                0 <= c['min_obstacle_z'] < c['max_obstacle_z'] and
                c['goal_end_margin'] >= 0 and c['goal_close_distance'] >= 0 and
                self.min_obstacle_points >= 1 and self.min_valid_points >= 1 and
                self.block_confirm_scans >= 1 and self.clear_confirm_scans >= 1 and
                self.cloud_timeout > 0 and self.pose_timeout > 0 and
                self.status_rate > 0 and self.debug_rate > 0):
            raise ValueError('Invalid obstacle_detector parameters')

        self.lock = threading.RLock()
        self.base_pose = None         # (world x, world y, world yaw)，来自既有定位话题
        self.pose_at = None
        self.goal = None              # (seq, world x, world y)
        self.cloud_at = None          # 已处理一帧点云的 monotonic 时间
        self.scan_goal_seq = None
        self.scan_result = 'FAULT_WAIT_SCAN'
        self.blocked = False
        self.clear_confirmed = False
        self.block_streak = 0
        self.clear_streak = 0
        self.count = 0
        self.nearest = math.nan
        self.diag = dict(front=0, path=0, height=0, self_removed=0,
                         nearest_front=math.nan, nearest_path=math.nan,
                         nearest_height=math.nan, nearest=math.nan, goal_distance=0.)

        self.pub = rospy.Publisher(self.status_topic, String, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, String, queue_size=5)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(self.goal_topic, String, self.on_goal, queue_size=1)
        rospy.Subscriber(self.lidar_topic, lidar_cls, self.on_cloud,
                         queue_size=1, buff_size=8 * 1024 * 1024)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.status_loop, daemon=True)
        self.thread.start()
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo('Obstacle ROI forward [%.2f,%.2f]m width %.2fm z[%.2f,%.2f]m; '
                      'lidar XYZ=(%.3f,%.3f,%.3f) RPY=(%.3f,%.3f,%.3f), calibrated=%s',
                      c['min_forward'], c['max_forward'], 2*c['corridor_half_width'],
                      c['min_obstacle_z'], c['max_obstacle_z'],
                      self.tx, self.ty, self.tz, roll, pitch, yaw,
                      self.extrinsics_calibrated)
        if not self.extrinsics_calibrated:
            rospy.logerr('MID360 foot-frame XYZ not calibrated. '
                         'Edit obstacle_params.yaml and set extrinsics_calibrated: true '
                         'AFTER measuring. Navigation will remain stopped.')

    def on_pose(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.expected_world_frame:
            rospy.logwarn_throttle(5.0, 'Unexpected pose frame_id %r',
                                   msg.header.frame_id)
            return
        q = msg.pose.orientation
        values = (q.x, q.y, q.z, q.w, msg.pose.position.x, msg.pose.position.y)
        if not all(math.isfinite(v) for v in values):
            return
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-8:
            return
        yaw = euler_from_quaternion((q.x/norm, q.y/norm, q.z/norm, q.w/norm))[2]
        with self.lock:
            self.base_pose = (float(msg.pose.position.x), float(msg.pose.position.y), yaw)
            self.pose_at = time.monotonic()

    def on_goal(self, msg):
        try:
            seq_s, x_s, y_s = msg.data.split('|')
            goal = (int(seq_s), float(x_s), float(y_s))
            if goal[0] <= 0 or not all(math.isfinite(v) for v in goal[1:]):
                raise ValueError('Bad goal')
        except (ValueError, TypeError):
            rospy.logwarn_throttle(5.0, 'Invalid obstacle_target message')
            return
        with self.lock:
            if self.goal != goal:
                self.block_streak = self.clear_streak = 0
                self.blocked = self.clear_confirmed = False
                self.scan_goal_seq = None
                self.scan_result = 'FAULT_WAIT_SCAN'
                self.count = 0
                self.nearest = math.nan
            self.goal = goal

    def on_cloud(self, msg):
        now = time.monotonic()
        with self.lock:
            pose, pose_at, goal = self.base_pose, self.pose_at, self.goal
        if pose is None or goal is None or pose_at is None or now - pose_at > self.pose_timeout:
            return

        pts = np.asarray([(p.x, p.y, p.z) for p in msg.points], dtype=np.float64)
        if pts.size == 0:
            pts = pts.reshape((0, 3))
        else:
            pts = pts[np.isfinite(pts).all(axis=1) &
                      (np.sum(pts*pts, axis=1) > 0.05**2)]

        # 新鲜点云不等于有效点云：点数不足时返回 FAULT 而不是 CLEAR。
        if len(pts) < self.min_valid_points:
            with self.lock:
                if self.goal != goal:
                    return
                self.cloud_at = now
                self.scan_goal_seq = goal[0]
                self.scan_result = 'FAULT_SPARSE_CLOUD'
                self.count = 0
                self.nearest = math.nan
                self.block_streak = self.clear_streak = 0
                self.clear_confirmed = False
            return

        # 注意：数据从 lidar -> foot，完全由 obstacle_params.yaml 提供的固定外参决定。
        foot = pts @ self.sensor_R.T + self.sensor_t

        # 使用导航所依赖的位姿作为世界系平面参考；若它不是足部原点，
        # 把 pose_origin_to_foot_x/y 经世界 yaw 旋转后加到定位 XY。
        bx, by, base_yaw = pose
        cy, sy = math.cos(base_yaw), math.sin(base_yaw)
        fx = bx + cy*self.pose_origin_to_foot_x - sy*self.pose_origin_to_foot_y
        fy = by + sy*self.pose_origin_to_foot_x + cy*self.pose_origin_to_foot_y
        dx = goal[1] - fx
        dy = goal[2] - fy
        goal_foot = (cy*dx + sy*dy, -sy*dx + cy*dy)

        n, diag = filter_points(foot, goal_foot, self.cfg)
        with self.lock:
            if self.goal != goal:
                return  # 处理期间人工切换目标：旧结果直接丢弃
            self.cloud_at = now
            self.scan_goal_seq = goal[0]
            self.count = n
            self.nearest = diag['nearest']
            self.diag = diag
            if n >= self.min_obstacle_points:
                self.block_streak += 1
                self.clear_streak = 0
                self.clear_confirmed = False
                if self.block_streak >= self.block_confirm_scans:
                    self.blocked = True
                self.scan_result = ('BLOCKED' if self.blocked else
                                    'FAULT_CONFIRMING_BLOCK')
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
        now = time.monotonic()
        with self.lock:
            goal_seq = self.goal[0] if self.goal else 0
            if not self.extrinsics_calibrated:
                state = 'FAULT_EXTRINSICS_NOT_CALIBRATED'
            elif self.goal is None:
                state = 'FAULT_NO_GOAL'
            elif self.base_pose is None or self.pose_at is None or now - self.pose_at > self.pose_timeout:
                state = 'FAULT_POSE_TIMEOUT'
            elif self.cloud_at is None or now - self.cloud_at > self.cloud_timeout:
                state = 'FAULT_LIDAR_TIMEOUT'
            elif self.scan_goal_seq != goal_seq:
                state = 'FAULT_WAIT_SCAN'
            else:
                state = self.scan_result
            return '{}|goal={}|points={}|nearest={:.3f}'.format(
                state, goal_seq, self.count, self.nearest)

    def debug_payload(self):
        with self.lock:
            d = dict(self.diag)
            count = self.count
            goal_seq = self.goal[0] if self.goal else 0
        return ('goal={} front={} path={} height={} self_removed={} accepted={} '
                'nearest_front={:.3f} nearest_path={:.3f} nearest_height={:.3f} '
                'nearest_accepted={:.3f} goal_dist={:.3f}').format(
            goal_seq, d['front'], d['path'], d['height'], d['self_removed'], count,
            d['nearest_front'], d['nearest_path'], d['nearest_height'],
            d['nearest'], d['goal_distance'])

    def status_loop(self):
        period = 1.0 / self.status_rate
        debug_period = 1.0 / self.debug_rate
        last_debug = 0.0
        last_state = None
        while not rospy.is_shutdown() and not self.stop_flag.is_set():
            status = self.status_payload()
            self.pub.publish(String(data=status))
            state = status.split('|', 1)[0]
            if state != last_state:
                rospy.loginfo('Obstacle detector => %s', status)
                last_state = state
            now = time.monotonic()
            if now - last_debug >= debug_period:
                self.debug_pub.publish(String(data=self.debug_payload()))
                last_debug = now
            self.stop_flag.wait(period)

    def on_shutdown(self):
        self.stop_flag.set()


if __name__ == '__main__':
    try:
        ObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
