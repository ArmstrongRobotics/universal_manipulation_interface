import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from xarm.wrapper import XArmAPI


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class RobotiqController(mp.Process):
    """
    Robotiq gripper controller using XArmAPI for Robotiq grippers attached to xArm robots.
    The Robotiq gripper is controlled through the xArm's gripper interface.
    Supports waypoint scheduling, trajectory interpolation, and real-time control.
    """
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname="10.11.12.103",
            port=1000,  # Not used for xArm API, kept for compatibility
            frequency=30,
            home_to_open=True,
            move_max_speed=200.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            verbose=False
            ):
        super().__init__(name="RobotiqController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
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

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RobotiqController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
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
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= robotiq specific methods ============
    def activate(self):
        # Activate the Robotiq gripper
        code, response = self.arm.robotiq_set_activate(wait=True, timeout=3)
        return code == 0

    def open(self, speed=255, force=255, wait=True):
        # Open Robotiq gripper to open position (0)
        code, response = self.arm.robotiq_open(speed=speed, force=force, wait=wait)
        return code == 0

    def close_gripper(self, speed=255, force=255, wait=True):
        # Close Robotiq gripper to close position (255)
        code, response = self.arm.robotiq_close(speed=speed, force=force, wait=wait)
        return code == 0

    def goto(self, pos, speed=255, force=255, wait=True):
        # Set Robotiq gripper to specific position (0-255 range)
        code, response = self.arm.robotiq_set_position(pos=pos, speed=speed, force=force, wait=wait)
        return code == 0

    def get_position(self):
        # Get current Robotiq gripper position from register 0x07D2
        code, response = self.arm.robotiq_get_status(number_of_registers=3)
        if code == 0 and len(response) >= 3:
            position = response[2] & 0xFF
            return position
        return 0

    def get_status(self):
        # Get full Robotiq gripper status
        code, response = self.arm.robotiq_get_status(number_of_registers=3)
        return code, response

    def reset(self):
        # Reset the Robotiq gripper
        code, response = self.arm.robotiq_reset()
        return code == 0
    
    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            self.arm = XArmAPI(self.hostname)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            
            # Initialize and activate gripper
            self.arm.robotiq_reset()
            time.sleep(0.5)
            self.arm.robotiq_set_activate(wait=True, timeout=3)
            
            # Home gripper
            if self.home_to_open:
                self.arm.robotiq_open(speed=255, force=255, wait=True)
                curr_pos = 0.0  # Open position
            else:
                self.arm.robotiq_close(speed=255, force=255, wait=True)
                curr_pos = 255.0  # Close position

            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos, 0, 0, 0, 0, 0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            last_pos = curr_pos
            
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                
                # Send position command if position changed significantly
                if abs(target_pos - last_pos) > 1.0:  # Position tolerance
                    self.arm.robotiq_set_position(
                        pos=int(target_pos), 
                        speed=min(255, int(abs(target_vel) * 10)), 
                        force=255, 
                        wait=False
                    )
                    last_pos = target_pos

                # Get state from gripper
                code, response = self.arm.robotiq_get_status(number_of_registers=3)
                if code == 0 and len(response) >= 3:
                    # Parse response registers
                    status_reg = response[0]  # Register 0x07D0
                    fault_reg = response[1]   # Register 0x07D1
                    pos_current_reg = response[2]  # Register 0x07D2
                    
                    position = pos_current_reg & 0xFF  # Position in lower byte
                    current = (pos_current_reg >> 8) & 0xFF  # Current in upper byte
                    
                    # Estimate velocity from position change
                    velocity = (position - getattr(self, '_last_position', position)) * self.frequency
                    self._last_position = position
                    
                    state = {
                        'gripper_state': status_reg,
                        'gripper_position': position / self.scale,
                        'gripper_velocity': velocity / self.scale,
                        'gripper_force': current,  # Use current as force approximation
                        'gripper_measure_timestamp': time.time(),
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency
                    }
                else:
                    # Fallback state if communication fails
                    state = {
                        'gripper_state': 0,
                        'gripper_position': 0.0,
                        'gripper_velocity': 0.0,
                        'gripper_force': 0.0,
                        'gripper_measure_timestamp': time.time(),
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency
                    }
                
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
            
        finally:
            if hasattr(self, 'arm'):
                self.arm.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[RobotiqController] Disconnected from robot: {self.hostname}")
