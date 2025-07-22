from lerobot.motors.motors_bus import MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus, Motor

norm_mode_body = MotorNormMode.DEGREES

# ✅ Correct calibration dictionary
calibration = {
    "shoulder_pan": MotorCalibration(
        id=1,
        drive_mode=0,
        homing_offset=2048,
        range_min=0,
        range_max=4095
    )
}

bus = FeetechMotorsBus(
    port="/dev/tty.usbmodem58FA1019951",
    motors={
        "shoulder_pan": Motor(1, "sts3215", norm_mode_body)
    },
    calibration=calibration
)

bus.connect()
print("Connected to motor bus.")

position = bus.sync_read("Present_Position")
print("Position:", position)

goal_pos = {"shoulder_pan": 0}  # example value in normalized scale
bus.sync_write("Goal_Position", goal_pos)

# Optional: add a small delay and disconnect
import time
time.sleep(1)
bus.disconnect(disable_torque=True)
print("Disconnected.")
