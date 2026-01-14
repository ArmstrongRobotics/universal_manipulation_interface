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

class XArmInterpolationController(mp.Process):
    """
    Interpolation controller for xArm robots using TCP/IP.
    Sends smooth pose trajectories to the robot.
    """
    def __init__(self,
            shm_manager,
            robot_ip,
            frequency=100,
            max_pos_speed=0.25,
            max_rot_speed=0.16,
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
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        # build ring buffer (minimal state)
        example = {
            'actual_pose': np.zeros((6,), dtype=np.float64),
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
            'target_time': target_time
        }
        self.input_queue.put(message)
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    def get_all_state(self):
        return self.ring_buffer.get_all()
    def run(self):
        arm = XArmAPI(self.robot_ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        if self.joints_init is not None:
            arm.move_joint(self.joints_init, speed=self.joints_init_speed, wait=True)
        dt = 1. / self.frequency
        curr_pose = np.array(arm.get_position(is_radian=True)[0:6])
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose]
        )
        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        while keep_running:
            t_now = time.monotonic()
            pose_command = pose_interp(t_now)
            # Send interpolated pose to xArm
            arm.set_position(*pose_command, speed=self.max_pos_speed, wait=False)
            # update robot state
            state = {
                'actual_pose': np.array(arm.get_position(is_radian=True)[0:6]),
                'robot_timestamp': time.time() - self.receive_latency
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
