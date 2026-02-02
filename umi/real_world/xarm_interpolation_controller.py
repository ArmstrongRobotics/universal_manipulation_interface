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
from umi.real_world.real_inference_util import (pose_to_mat,
                                                mat_to_pose 
                                                )
from scipy.spatial.transform import Rotation as R

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
            frequency=30.0,
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
        self.step_event = mp.Event()  # Event to gate each servo step
        self.step_event_trigger = mp.Event()
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
        arm_pose = np.array(pose[:6])
        assert arm_pose.shape == (6,)
        gripper_pos = pose[6]
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': arm_pose,
            'target_time': target_time,
            'gripper_pos': gripper_pos
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
        arm.set_mode(7)
        arm.set_state(0)
        
        # Initialize gripper
        print("[XArmInterpolationController] Initializing Robotiq gripper...")
        arm.robotiq_get_status()
        gripper_was_already_activated = arm.robotiq_status.get("gSTA", 0) == 3
        if gripper_was_already_activated:
            print("Gripper activated, will open...")
        else:
            raise RuntimeError('Gripper not activated.')
        
        # Home gripper to open position
        print("[XArmInterpolationController] Homing gripper to open...")
        arm.robotiq_open(speed=0xFF, force=0x32, wait=True)
        curr_gripper_pos = .051  # Open position in meters
        last_gripper_pos = curr_gripper_pos
        
        dt = 1. / self.frequency
        print("Printing initial robot pose:")
        ret = arm.get_position(is_radian=True)
        if ret[0] != 0:
            raise RuntimeError(f"Failed to get initial position, error code: {ret[0]}")
        curr_pose = np.array(ret[1][0:6])

        curr_pose[:3] = curr_pose[:3] / 1000.0  # convert mm to meters
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
            ret = arm.get_position(is_radian=True)
            if ret[0] != 0:
                raise RuntimeError(f"Failed to get initial position, error code: {ret[0]}")
            pose_command = pose_interp(t_now)

            # Calculate difference between pose_command and curr_pose
            pos_diff = pose_command[:3] - curr_pose[:3]
            rot_diff_rad = pose_command[3:6] - curr_pose[3:6]
            rot_diff_deg = np.degrees(rot_diff_rad)

            # Print difference between pose_command and curr_pose
            curr_pose_mat = pose_to_mat(curr_pose)
            pose_command_mat = pose_to_mat(pose_command)
            new_pose_in_cur_pose = np.linalg.inv(curr_pose_mat) @ pose_command_mat
            new_pose_in_cur_pose_pose = mat_to_pose(new_pose_in_cur_pose)
            # Convert last 3 elements (rotation vector) to degrees for clarity
            rotvec_rad = new_pose_in_cur_pose_pose[3:6]
            rotvec_deg = np.degrees(rotvec_rad)
            # Only wait for event if any rot diff deg > 1 or any pos diff > 0.01
            if np.any(np.abs(rotvec_deg) > 5.0) or np.any(np.abs(pos_diff) > 0.05):
                self.step_event_trigger.set()
                print("Waiting on input from user")
                print("Pose command:", pose_command)
                print("Current pose:", curr_pose)
                print("Pose diff: pos (m):", pos_diff, "rot (deg):", rot_diff_deg)
                print("Pose diff in deg via new_pose_in_cur_pose:", rotvec_deg)
                self.step_event.wait()
                self.step_event.clear()
                self.step_event_trigger.clear()

            # Send interpolated pose to xArm
            pose_command_mm = pose_command.copy()
            pose_command_mm[:3] = pose_command_mm[:3] * 1000.  # convert to mm
            pose_command_mm_aa = pose_command_mm.copy()
            # Convert euler angles to axis-angle for xArm API
            r = R.from_euler('xyz', pose_command_mm[3:6], degrees=False)
            axis_angle = r.as_rotvec()
            pose_command_mm_aa[3:6] = axis_angle
            arm.set_servo_cartesian_aa(pose_command_mm_aa, speed=self.max_pos_speed, mvacc=None, mvtime=0, is_radian=True)
            # Handle gripper control
            dt = 1 / self.frequency
            gripper_target_pos = gripper_interp(t_now)[0]
            gripper_target_vel = (gripper_interp(t_now)[0] - gripper_interp(t_now - dt)[0]) / dt

            gripper_target_pos = np.clip(gripper_target_pos, 0.0, 0.051)  # Gripper range in meters
            # # Map to Robotiq position value (0-255)
            gripper_target_pos_robotiq = int(np.clip(255 * (1 - (gripper_target_pos / 0.051)), 0, 255))
            # # Send gripper command if position changed significantly
            if abs(gripper_target_pos - last_gripper_pos) > .001:  # Position tolerance
                arm.robotiq_set_position(
                    pos=gripper_target_pos_robotiq, 
                    speed=min(255, int(abs(gripper_target_vel) * 10)), 
                    force=255, 
                    wait=False
                )
                last_gripper_pos = gripper_target_pos
            # update robot state
            ret = arm.get_position(is_radian=True)
            if ret[0] != 0:
                raise RuntimeError(f"Failed to get position, error code: {ret[0]}")
            curr_pose = np.array(ret[1][0:6])
            curr_pose[:3] = curr_pose[:3] / 1000.0  # convert mm to meters
            # Get joint positions and velocities
            joint_ret = arm.get_joint_states()
            if joint_ret[0] != 0:
                raise RuntimeError(f"Failed to get joint states, error code: {joint_ret[0]}")
        
            
            # Note: xArm API doesn't directly provide TCP velocity, so we'll use zeros
            actual_tcp_speed = np.zeros(6)
            
            # Get gripper state
            code, response = arm.robotiq_get_status(number_of_registers=3)
            if code != 0 and len(response) < 3:
                raise RuntimeError(f"Gripper status response error: code={code}, response={response}")
            # Parse response registers
            status_reg = response[0]  # Register 0x07D0
            fault_reg = response[1]   # Register 0x07D1
            pos_current_reg = response[2]  # Register 0x07D2
            
            gripper_position_raw = pos_current_reg & 0xFF  # Position in lower byte
            # # Map 0 to 0.051m (open), 251 to 0m (closed)
            gripper_position = np.clip(0.051 * (1 - (gripper_position_raw / 255.0)), 0.0, 0.051)
            gripper_current = (pos_current_reg >> 8) & 0xFF  # Current in upper byte

            state = {
                'ActualTCPPose': curr_pose,
                'ActualTCPSpeed': actual_tcp_speed,
                'ActualQ': None,
                'ActualQd': None,
                'TargetTCPPose': pose_command,
                'TargetTCPSpeed': np.zeros(6),  # Could be computed from trajectory
                'TargetQ': np.zeros(7),  # Could be computed via IK if needed
                'TargetQd': np.zeros(7),  # Could be computed from trajectory
                'robot_timestamp': time.time() - self.receive_latency,
                'gripper_state': 0,
                'gripper_position': .051,
                'gripper_velocity': 0,
                'gripper_force': 0,  # Use current as force approximation
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
