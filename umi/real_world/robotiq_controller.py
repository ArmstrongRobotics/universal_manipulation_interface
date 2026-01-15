import time
from xarm.wrapper import XArmAPI

class RobotiqController:
    """
    Robotiq gripper controller using XArmAPI for Robotiq grippers attached to xArm robots.
    The Robotiq gripper is controlled through the xArm's gripper interface.
    Supports open, close, set position, and read width.
    """
    def __init__(self, ip="10.11.12.103", timeout=2.0):
        self.ip = ip
        self.timeout = timeout
        self.arm = XArmAPI(self.ip)
        self.connect()

    def connect(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def close(self):
        if self.arm:
            self.arm.disconnect()
            self.arm = None

    def activate(self):
        # Activate the Robotiq gripper
        code, response = self.arm.robotiq_set_activate(wait=True, timeout=3)
        return code == 0  # Return True if successful

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
        # 0 = open, 255 = closed
        code, response = self.arm.robotiq_set_position(pos=pos, speed=speed, force=force, wait=wait)
        return code == 0

    def get_position(self):
        # Get current Robotiq gripper position from register 0x07D2
        code, response = self.arm.robotiq_get_status(number_of_registers=3)
        if code == 0 and len(response) >= 3:
            # Position is in the third register (0x07D2)
            # Extract position from the response
            position = response[2] & 0xFF  # Position is in lower byte
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

    def __del__(self):
        self.close()
