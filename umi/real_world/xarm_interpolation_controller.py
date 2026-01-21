import os
import time
import enum
import multiprocessing as mp
import numpy as np
from xarm.wrapper import XArmAPI
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1
    # SCHEDULE_GRIPPER_WAYPOINT removed

class XArmInterpolationController(mp.Process):
    """
    Interpolation controller for xArm robots using TCP/IP.
    Sends smooth pose trajectories to the robot and controls Robotiq gripper.
    """
    def __init__(self,
            shm_manager,
            robot_ip,
            frequency=100,
            max_pos_speed=0.05,
            max_rot_speed=0.16,
            gripper_max_speed=200.0,
            launch_timeout=3,
            joints_init=None,
            joints_init_speed=1.0,
            verbose=False,
            get_max_k=None,
            receive_latency=0.0):
        super().__init__(name="XArmInterpolationController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.gripper_max_speed = gripper_max_speed
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.receive_latency = receive_latency
        self.verbose = verbose
        if get_max_k is None:
            get_max_k = int(frequency * 5)
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_time': 0.0,
            'gripper_pos': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        # build ring buffer (full state matching RTDE controller + gripper state)
        example = {
            'ActualTCPPose': np.zeros((6,), dtype=np.float64),
            'ActualTCPSpeed': np.zeros((6,), dtype=np.float64),
            'ActualQ': np.zeros((7,), dtype=np.float64),  # 7 joints for xArm
            'ActualQd': np.zeros((7,), dtype=np.float64),
            'TargetTCPPose': np.zeros((6,), dtype=np.float64),
            'TargetTCPSpeed': np.zeros((6,), dtype=np.float64),
            'TargetQ': np.zeros((7,), dtype=np.float64),
            'TargetQd': np.zeros((7,), dtype=np.float64),
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[XArmInterpolationController] Controller process spawned at {self.pid}")
    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    def stop_wait(self):
        self.join()
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time,
            'gripper_pos': 0.0  # Default gripper position
        }
        self.input_queue.put(message)

    # schedule_gripper_waypoint removed
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    def get_all_state(self):
        return self.ring_buffer.get_all()
    def run(self):
        print("[XArmInterpolationController] Starting controller process with ip:", self.robot_ip)
        arm = XArmAPI(self.robot_ip)
        arm.motion_enable(enable=True)
        arm.set_mode(1)
        arm.set_state(0)
        
        # Initialize gripper
        print("[XArmInterpolationController] Initializing Robotiq gripper...")
        arm.robotiq_get_status()
        gripper_was_already_activated = arm.robotiq_status.get("gSTA", 0) == 3
        if gripper_was_already_activated:
            print("Gripper activated, will open...")
        else:
            print("Gripper NOT activated, activating temporarily to open and then will deactivate again")
            raise RuntimeError("Robotiq gripper must be activated before use.")
        
        # Home gripper to open position
        print("[XArmInterpolationController] Homing gripper to open...")
        arm.robotiq_open(speed=0xFF, force=0x32, wait=True)
        curr_gripper_pos = 0.0  # Open position
        last_gripper_pos = curr_gripper_pos
        
        dt = 1. / self.frequency
        print("Printing initial robot pose:")
        ret = arm.get_position(is_radian=True)
        if ret[0] != 0:
            raise RuntimeError(f"Failed to get initial position, error code: {ret[0]}")
        curr_pose = np.array(ret[1][0:6])
        print(curr_pose)
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose]
        )
        # Initialize gripper trajectory interpolation
        gripper_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[[curr_gripper_pos, 0, 0, 0, 0, 0]]  # Only position matters for gripper
        )
        last_gripper_waypoint_time = curr_t
        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        while keep_running:
            t_now = time.monotonic()
            pose_command = pose_interp(t_now)
            # Send interpolated pose to xArm
            arm.set_servo_cartesian(pose_command, speed=self.max_pos_speed, mvacc=None, mvtime=0, is_radian=True)
            
            # Handle gripper control
            dt = 1 / self.frequency
            gripper_target_pos = gripper_interp(t_now)[0]
            gripper_target_vel = (gripper_interp(t_now)[0] - gripper_interp(t_now - dt)[0]) / dt
            
            # Send gripper command if position changed significantly
            if abs(gripper_target_pos - last_gripper_pos) > 1.0:  # Position tolerance
                arm.robotiq_set_position(
                    pos=int(gripper_target_pos), 
                    speed=min(255, int(abs(gripper_target_vel) * 10)), 
                    force=255, 
                    wait=False
                )
                last_gripper_pos = gripper_target_pos
            # update robot state
            ret = arm.get_position(is_radian=True)
            if ret[0] != 0:
                raise RuntimeError(f"Failed to get position, error code: {ret[0]}")
            
            # Get joint positions and velocities
            joint_ret = arm.get_joint_states()
            if joint_ret[0] != 0:
                raise RuntimeError(f"Failed to get joint states, error code: {joint_ret[0]}")
            
            actual_joints = np.array(joint_ret[1][0])  # joint positions
            actual_joint_vels = np.array(joint_ret[1][1])  # joint velocities
            
            # Get TCP velocity (if available, otherwise estimate or use zeros)
            actual_tcp_pose = np.array(ret[1][0:6])
            # Note: xArm API doesn't directly provide TCP velocity, so we'll use zeros
            # In a real implementation, you might estimate this from pose differences
            actual_tcp_speed = np.zeros(6)
            
            # Get gripper state
            code, response = arm.robotiq_get_status(number_of_registers=3)
            if code != 0 and len(response) < 3:
                raise RuntimeError(f"Gripper status response error: code={code}, response={response}")
            # Parse response registers
            status_reg = response[0]  # Register 0x07D0
            fault_reg = response[1]   # Register 0x07D1
            pos_current_reg = response[2]  # Register 0x07D2
            
            gripper_position = pos_current_reg & 0xFF  # Position in lower byte
            gripper_current = (pos_current_reg >> 8) & 0xFF  # Current in upper byte
            
            # Estimate velocity from position change
            gripper_velocity = (gripper_position - getattr(self, '_last_gripper_position', gripper_position)) * self.frequency
            self._last_gripper_position = gripper_position

            state = {
                'ActualTCPPose': actual_tcp_pose,
                'ActualTCPSpeed': actual_tcp_speed,
                'ActualQ': actual_joints,
                'ActualQd': actual_joint_vels,
                'TargetTCPPose': pose_command,
                'TargetTCPSpeed': np.zeros(6),  # Could be computed from trajectory
                'TargetQ': np.zeros(7),  # Could be computed via IK if needed
                'TargetQd': np.zeros(7),  # Could be computed from trajectory
                'robot_timestamp': time.time() - self.receive_latency,
                'gripper_state': status_reg,
                'gripper_position': gripper_position / 1000.0,  # Convert to meters if needed
                'gripper_velocity': gripper_velocity / 1000.0,
                'gripper_force': gripper_current,  # Use current as force approximation
                'gripper_measure_timestamp': time.time(),
                'gripper_receive_timestamp': time.time(),
                'gripper_timestamp': time.time() - self.receive_latency
            }
            self.ring_buffer.put(state)
            # fetch command from queue
            try:
                commands = self.input_queue.get_k(1)
                n_cmd = len(commands['cmd'])
            except Empty:
                n_cmd = 0
            for i in range(n_cmd):
                command = dict()
                for key, value in commands.items():
                    command[key] = value[i]
                cmd = command['cmd']
                if cmd == Command.STOP.value:
                    keep_running = False
                    break
                elif cmd == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    target_time = float(command['target_time'])
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_waypoint_time
                    )
                    last_waypoint_time = target_time
                    gripper_pos = float(command['gripper_pos'])
                    gripper_interp = gripper_interp.schedule_waypoint(
                        pose=[gripper_pos, 0, 0, 0, 0, 0],
                        time=target_time,
                        max_pos_speed=self.gripper_max_speed,
                        max_rot_speed=self.gripper_max_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_gripper_waypoint_time
                    )
                    last_gripper_waypoint_time = target_time
                else:
                    keep_running = False
                    break
            t_wait_util = t_start + (iter_idx + 1) * dt
            precise_wait(t_wait_util, time_func=time.monotonic)
            if iter_idx == 0:
                self.ready_event.set()
            iter_idx += 1
        arm.disconnect()
        self.ready_event.set()
        if self.verbose:
            print(f"[XArmInterpolationController] Disconnected from robot: {self.robot_ip}")
