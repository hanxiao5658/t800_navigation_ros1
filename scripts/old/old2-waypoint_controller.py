#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T800 顺序 Waypoint 导航控制器（ROS1 Noetic）。

【导航流程】
    start 服务 -> ALIGN_HEADING（先原地朝向目标）
    -> MOVE_FORWARD（直线前进，允许小角度航向修正）
    -> ALIGN_FINAL_YAW（到点原地转到 YAML 指定 yaw）
    -> WAIT_FOR_NEXT（原地停车，必须人工确认才继续）
    行进中：遇障 -> STOP_FOR_OBSTACLE；检测数据失效 -> WAIT_FOR_OBSTACLE_DATA
    -> next_waypoint 服务 -> 下一个 Waypoint
    最后一个 Waypoint 到达并完成转向 -> FINISHED（停车）

【输入话题】
    /my_ros1/navigation/base_pose_gt : geometry_msgs/PoseStamped
    （Gazebo 真值；后续可替换为已经对齐坐标系的 FAST-LIO 位姿）

【新增检测输入】
    /my_ros1/navigation/obstacle_status : std_msgs/String
    由独立节点以固定频率发布 CLEAR/BLOCKED/FAULT_*，附带目标序号。
    控制器同时检查状态和心跳：只有新鲜 CLEAR 才允许运动。
【输出话题，均以 /my_ros1 开头】
    /my_ros1/motion/body_vel_cmd          : geometry_msgs/Twist，机体坐标系速度
    /my_ros1/navigation/state            : std_msgs/String，导航状态
    /my_ros1/navigation/active_waypoint  : geometry_msgs/PoseStamped，当前目标
    /my_ros1/navigation/obstacle_target  : std_msgs/String，供检测节点使用的原子目标 ID 与 XY
    /my_ros1/navigation/waypoints_path   : nav_msgs/Path，预设路径（可视化）

【控制服务（也以 /my_ros1 开头）】
    /my_ros1/navigation/start            : std_srvs/Trigger，开始/恢复
    /my_ros1/navigation/pause            : std_srvs/Trigger，暂停
    /my_ros1/navigation/next_waypoint    : std_srvs/Trigger，人工确认下一点

注意：
1. YAML waypoint 的 yaw 是到达位置后的最终朝向，不是行走时的朝向。
2. 旧 YAML 的 wait 字段不再生效；等待时长由人工决定。
3. 障碍物检测独立节点处理；本控制器不解析点云，仅负责心跳/状态门控。
4. 下游接口无超时看门狗。本节点正常运行时会在停车状态持续发零速度；
   Ctrl+C 时也尝试发送多次零速度，但跨 ROS Bridge/LCM 不保证送达。
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
    """将航向误差限制在 [-pi, pi]，避免在 ±180° 附近转大圈。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def saturated(value, limit):
    """把带符号速度限制在 [-limit, limit]。"""
    return max(-limit, min(limit, value))


def signed_min_magnitude(value, minimum):
    """非零指令小于最低速度时提升到最低速度，以克服步态死区。"""
    if value == 0.0:
        return 0.0
    return math.copysign(max(abs(value), minimum), value)


class WaypointController:
    def __init__(self):
        rospy.init_node('waypoint_controller')

        # 沿用之前 navigation_params.yaml 的参数名和默认值，避免覆盖你的调参结果。
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

        # 障碍物检测心跳：不受 Gazebo /clock 暂停影响，使用 monotonic 时间。
        # enabled=false 仅用于隔离诊断；正常导航应一直开启。
        self.obstacle_enabled = bool(rospy.get_param('~obstacle_enabled', True))
        self.obstacle_topic = rospy.get_param(
            '~obstacle_topic', '/my_ros1/navigation/obstacle_status')
        self.obstacle_timeout = float(rospy.get_param('~obstacle_timeout', 0.5))
        self.safety_poll_rate = float(rospy.get_param('~safety_poll_rate', 20.0))
        if self.obstacle_timeout <= 0.0 or self.safety_poll_rate <= 0.0:
            raise ValueError('obstacle_timeout and safety_poll_rate must be > 0')

        if self.rate <= 0.0 or self.pose_timeout <= 0.0:
            raise ValueError('control_rate and pose_timeout must be > 0')
        if not (0.0 < self.yaw_tol < self.realign < math.pi):
            raise ValueError('yaw_tolerance_deg must be < realign_threshold_deg < 180')
        if self.xy_tol <= 0.0 or self.min_forward < 0.0 or self.max_forward <= 0.0:
            raise ValueError('Invalid translation tolerance/speed')
        if self.min_forward > self.max_forward or self.min_turn > self.max_turn:
            raise ValueError('Minimum speed cannot exceed maximum speed')
        if min(self.kp_forward, self.kp_turn, self.kp_steer) < 0.0:
            raise ValueError('P gains must be nonnegative')

        # 兼容上一版 YAML，不要求你修改 waypoints.yaml。
        # wait 不再参与自动跳点；人工服务是唯一的下一点触发入口。
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
                }
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise ValueError('Invalid waypoint {}: {}'.format(i, exc))
            if not all(math.isfinite(point[k]) for k in ('x', 'y', 'yaw')):
                raise ValueError('Non-finite waypoint {}'.format(i))
            self.waypoints.append(point)
        if any('wait' in p for p in raw):
            rospy.logwarn('Legacy waypoint "wait" fields are ignored; '
                          'advancement requires /my_ros1/navigation/next_waypoint')

        # ROS Subscriber、Timer、Service 回调可以在不同线程运行。
        # RLock 使状态改变和控制指令发布彼此互斥，防止按钮点击与控制周期竞争。
        self.lock = threading.RLock()
        self.pose = None  # (x, y, yaw)，Gazebo world 坐标系
        self.pose_wall_time = None  # 单调时钟，判断定位是否超时
        self.enabled = self.autostart
        self.index = 0  # 当前正在执行的 waypoint 索引（从 0 开始）
        self.state = 'WAIT_FOR_POSE' if self.autostart else 'IDLE'
        self.paused_at_waypoint = False  # 暂停前是否处于 WAIT_FOR_NEXT
        self.shutdown_requested = False
        # 状态取值：CLEAR / BLOCKED / FAULT_*；必须与当前目标序号匹配。
        self.obstacle_value = 'FAULT_NO_HEARTBEAT'
        self.obstacle_goal_seq = -1
        self.obstacle_wall_time = None

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            '/my_ros1/navigation/state', String, queue_size=5, latch=True)
        self.goal_pub = rospy.Publisher(
            '/my_ros1/navigation/active_waypoint', PoseStamped,
            queue_size=1, latch=True)
        self.path_pub = rospy.Publisher(
            '/my_ros1/navigation/waypoints_path', Path, queue_size=1, latch=True)
        self.obstacle_goal_pub = rospy.Publisher(
            '/my_ros1/navigation/obstacle_target', String,
            queue_size=1, latch=True)

        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(self.obstacle_topic, String, self.on_obstacle, queue_size=5)
        rospy.Service('/my_ros1/navigation/start', Trigger, self.on_start)
        rospy.Service('/my_ros1/navigation/pause', Trigger, self.on_pause)
        rospy.Service('/my_ros1/navigation/next_waypoint', Trigger, self.on_next_waypoint)
        rospy.on_shutdown(self.on_shutdown)

        self.publish_path()
        self.publish_goal()
        self.publish_status()
        rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self.control_step)
        # ROS1 开启 use_sim_time 时，rospy.Timer 会随 /clock 暂停；另起实时时钟
        # 看门狗线程，保证通信中断时仍然能够主动下发零速度。
        self.safety_thread = threading.Thread(target=self.safety_watchdog, daemon=True)
        self.safety_thread.start()
        rospy.loginfo('T800 navigator ready: %d waypoints; manual next enabled; autostart=%s',
                      len(self.waypoints), self.autostart)

    def on_pose(self, msg):
        """更新 Gazebo 真值位姿；不接受坐标系错误或无效数据。"""
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.logwarn_throttle(5.0, 'Pose frame %r differs from %r',
                                   msg.header.frame_id, self.frame_id)
            return
        q = msg.pose.orientation
        p = msg.pose.position
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-8 or not all(math.isfinite(v) for v in
                                  (p.x, p.y, q.x, q.y, q.z, q.w)):
            rospy.logwarn_throttle(5.0, 'Invalid ground-truth pose ignored')
            return
        yaw = euler_from_quaternion((q.x/norm, q.y/norm, q.z/norm, q.w/norm))[2]
        with self.lock:
            self.pose = (p.x, p.y, yaw)
            self.pose_wall_time = time.monotonic()

    def pose_is_fresh(self):
        """调用方需已经持有 self.lock（RLock 支持嵌套）。"""
        with self.lock:
            return (self.pose is not None and self.pose_wall_time is not None
                    and time.monotonic() - self.pose_wall_time <= self.pose_timeout)

    def on_obstacle(self, msg):
        """接收独立检测节点的固定频率状态和心跳。

        格式：CLEAR|goal=2|points=0|nearest=nan
        使用 goal 序号防止切换 Waypoint 后误采用上一个目标的 CLEAR。
        收到格式错误消息时按故障处理，而不是默认为安全。
        """
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
        if state != 'CLEAR' and state != 'BLOCKED' and not state.startswith('FAULT_'):
            state = 'FAULT_BAD_STATUS'
        with self.lock:
            self.obstacle_value = state
            self.obstacle_goal_seq = seq
            self.obstacle_wall_time = time.monotonic()

    def obstacle_health(self):
        """返回 CLEAR/BLOCKED/FAULT_*，仅新鲜且与当前目标匹配的 CLEAR 放行。

        由控制线程或服务回调持锁调用；本函数不访问 ROS 仿真时钟。
        """
        if not self.obstacle_enabled:
            return 'CLEAR'
        if (self.obstacle_wall_time is None or
                time.monotonic() - self.obstacle_wall_time > self.obstacle_timeout):
            return 'FAULT_COMM_TIMEOUT'
        if self.obstacle_goal_seq != self.index + 1:
            return 'FAULT_GOAL_MISMATCH'
        return self.obstacle_value

    def safety_watchdog(self):
        """真实时间线程：Gazebo 暂停或检测节点崩溃时，仍持续强制停车。

        注意：ROS1/ROS2 Bridge/LCM 也可能全部断链，因此此层不替代硬件急停。
        """
        interval = 1.0 / self.safety_poll_rate
        while not rospy.is_shutdown() and not self.shutdown_requested:
            with self.lock:
                moving_task = (self.enabled and self.state not in
                               ('WAIT_FOR_NEXT', 'FINISHED', 'IDLE'))
                if moving_task:
                    obs = self.obstacle_health()
                    if not self.pose_is_fresh():
                        self.change_state('WAIT_FOR_POSE')
                        self.publish_cmd(0.0, 0.0)
                    elif obs != 'CLEAR':
                        new_state = ('STOP_FOR_OBSTACLE' if obs == 'BLOCKED'
                                     else 'WAIT_FOR_OBSTACLE_DATA')
                        self.change_state(new_state)
                        self.publish_cmd(0.0, 0.0)
            time.sleep(interval)

    def on_start(self, _req):
        """开始/恢复导航；从 WAIT_FOR_NEXT 恢复时绝不跳过人工确认。"""
        with self.lock:
            if self.shutdown_requested:
                return TriggerResponse(success=False, message='Controller is shutting down')
            if self.enabled:
                if self.state == 'WAIT_FOR_NEXT':
                    return TriggerResponse(success=False,
                                           message='Waiting for operator: call next_waypoint, not start')
                return TriggerResponse(success=False, message='Navigation already running')
            if not self.pose_is_fresh():
                return TriggerResponse(success=False, message='No fresh ground-truth pose')
            # 开始任务时要求障碍物节点已在线且正在报告 CLEAR。
            if self.obstacle_enabled and self.obstacle_health() != 'CLEAR':
                return TriggerResponse(success=False,
                                       message='Obstacle detector not CLEAR: {}'.format(
                                           self.obstacle_health()))

            # 只有上一轮任务完成后再次 start，才从 wp1 开始新一轮。
            if self.state == 'FINISHED' or self.index >= len(self.waypoints):
                self.index = 0
                self.paused_at_waypoint = False

            self.enabled = True
            if self.paused_at_waypoint:
                # 原先在某个 waypoint 等待人工，暂停后重新 start 也必须继续等待。
                self.paused_at_waypoint = False
                self.change_state('WAIT_FOR_NEXT')
                message = 'Resumed; still waiting for next_waypoint at waypoint {}'.format(
                    self.index + 1)
            else:
                self.change_state('ALIGN_HEADING')
                message = 'Navigation started at waypoint {}'.format(self.index + 1)
            self.publish_goal()
            self.publish_cmd(0.0, 0.0)
            return TriggerResponse(success=True, message=message)

    def on_pause(self, _req):
        """暂停当前任务并停车，不删除目标点索引。"""
        with self.lock:
            if self.state == 'WAIT_FOR_NEXT':
                self.paused_at_waypoint = True
            self.enabled = False
            if self.state != 'FINISHED':
                self.change_state('IDLE')
            self.publish_cmd(0.0, 0.0)
            return TriggerResponse(success=True, message='Paused; zero velocity requested')

    def on_next_waypoint(self, _req):
        """模拟平板点击：只能在到点且等待人工时，放行下一个 waypoint。"""
        with self.lock:
            if self.shutdown_requested:
                return TriggerResponse(success=False, message='Controller is shutting down')
            if not self.enabled:
                return TriggerResponse(success=False, message='Navigation not running; call start first')
            if self.state != 'WAIT_FOR_NEXT':
                return TriggerResponse(success=False,
                                       message='Not waiting at waypoint (current state: {})'.format(self.state))
            if not self.pose_is_fresh():
                return TriggerResponse(success=False, message='No fresh ground-truth pose')
            if self.index >= len(self.waypoints) - 1:
                return TriggerResponse(success=False, message='No next waypoint')

            self.index += 1  # 按一次按钮，仅增加一个目标点
            # 新目标必须重新由检测节点处理；禁止沿用旧目标 CLEAR。
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
        """状态有变化时发布一次状态话题并写日志。"""
        if self.state != state:
            self.state = state
            self.publish_status()
            rospy.loginfo('Nav state => %s; waypoint %d/%d',
                          state, min(self.index + 1, len(self.waypoints)),
                          len(self.waypoints))

    def publish_status(self):
        label = self.waypoints[min(self.index, len(self.waypoints)-1)]['id']
        self.status_pub.publish(String(data='state={} waypoint={}/{} id={} obstacle={}'.format(
            self.state, min(self.index + 1, len(self.waypoints)),
            len(self.waypoints), label,
            self.obstacle_health() if self.obstacle_enabled else 'DISABLED')))

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
            # 不依赖 Header.seq（某些 ROS1 传输会自动重编号）；用一条消息
            # 原子地携带目标序号和坐标，避免旧目标的 CLEAR 被误用于新目标。
            self.obstacle_goal_pub.publish(String(
                data='{}|{:.9f}|{:.9f}'.format(
                    self.index + 1, wp['x'], wp['y'])))

    def publish_path(self):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = rospy.Time.now()
        path.poses = [self.as_pose(p) for p in self.waypoints]
        self.path_pub.publish(path)

    def publish_cmd(self, vx, wz):
        """只发布 BODY 坐标速度；下游直接映射至 BodyVelCmd。"""
        msg = Twist()
        msg.linear.x = float(vx)   # 机体前方为正（m/s）
        msg.angular.z = float(wz)  # 机体逆时针旋转为正（rad/s）
        self.cmd_pub.publish(msg)

    def turn_cmd(self, err):
        raw = saturated(self.kp_turn * err, self.max_turn)
        return saturated(signed_min_magnitude(raw, self.min_turn), self.max_turn)

    def control_step(self, _event):
        """固定频率运行的状态机；WAIT_FOR_NEXT 时只发布零速度。"""
        with self.lock:
            # IDLE、FINISHED、正在关闭时，始终保持零速度。
            if self.shutdown_requested or not self.enabled:
                self.publish_cmd(0.0, 0.0)
                return

            # 最高优先级的“人工等待”：不能因定位超时恢复而自行开始移动。
            if self.state == 'WAIT_FOR_NEXT':
                self.publish_cmd(0.0, 0.0)
                return

            # 行走/转向时如果真值丢失，立即停车；恢复后重新朝向当前点。
            if not self.pose_is_fresh():
                self.change_state('WAIT_FOR_POSE')
                self.publish_cmd(0.0, 0.0)
                return
            # 无论处于转向还是行走，检测节点失效或判定阻挡都立即停车。
            # 控制侧自身心跳超时与检测侧点云超时是两套独立的看门狗。
            obs = self.obstacle_health()
            if obs != 'CLEAR':
                self.change_state('STOP_FOR_OBSTACLE' if obs == 'BLOCKED'
                                  else 'WAIT_FOR_OBSTACLE_DATA')
                self.publish_cmd(0.0, 0.0)
                return
            if self.state in ('WAIT_FOR_POSE', 'STOP_FOR_OBSTACLE',
                              'WAIT_FOR_OBSTACLE_DATA'):
                # 故障恢复时先重新对准当前 Waypoint，不直接使用旧行走指令。
                self.change_state('ALIGN_HEADING')

            x, y, yaw = self.pose
            goal = self.waypoints[self.index]
            dx = goal['x'] - x
            dy = goal['y'] - y
            dist = math.hypot(dx, dy)

            if self.state == 'ALIGN_HEADING':
                # 位置已在容差内：无需先对准行走方向，直接调整最终 yaw。
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
                    # 航向偏差太大，停止平移并重新原地对准目标。
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
                    self.publish_cmd(0.0, 0.0)
                    if self.index == len(self.waypoints) - 1:
                        # 最后一个点没有“下一点”，直接结束任务。
                        self.enabled = False
                        self.change_state('FINISHED')
                        rospy.loginfo('Completed all %d waypoints', len(self.waypoints))
                    else:
                        # 到点后无限期等待平板/人工按钮，不再使用 YAML wait。
                        self.change_state('WAIT_FOR_NEXT')
                else:
                    self.publish_cmd(0.0, self.turn_cmd(err))

            else:
                # 未知状态不允许继续运动。
                rospy.logwarn_throttle(5.0, 'Unknown navigation state %r', self.state)
                self.publish_cmd(0.0, 0.0)

    def on_shutdown(self):
        """Ctrl+C 或 roslaunch 退出时尽力向下游发送停止命令。"""
        with self.lock:
            self.shutdown_requested = True
            self.enabled = False
        # ROS1 发布端/bridge/ROS2/LCM 可能同时关闭，以下仅保证尝试发送。
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
