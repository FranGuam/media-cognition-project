from pymycobot.mycobot import MyCobot
import time

# LEFT = 75
# RIGHT = -35
# NEAR = 135
# FAR = 215

BOX_LEVEL = 105
# BOX_LEVEL = 90
LEFT_BORDER = 80
RIGHT_BORDER = -70
NEAR_BORDER = 90
FAR_BORDER = 235

MODE = 0
DEFAULT_SPEED = 40
TARGET_DELTA = 20

POSITION = {
    "left-near":
        [[120, 170, 240, -180, 0, 0],  # up
         [120, 170, 175, -180, 0, 0]],  # down
    "left-far":
        [[200, 120, 230, -180, 0, 0],  # up
         [215, 150, 220, -150, 0, -60]],  # down
    "right-near":
        [[120, -150, 240, -180, 0, 0],  # up
         [120, -150, 200, -180, 0, 0]],  # down
    "right-far":
        [[200, -100, 235, -180, 0, 0],  # up
         [220, -140, 220, -150, -10, -80]],  # down
}

ANGLE = {
    "down": [180, 0, 180],
}

mc = MyCobot('COM3', 115200)


def cmp(pos, target, quite=False):
    if len(pos) == 0:
        return 999
    delta_x = abs(pos[0] - target[0])
    delta_y = abs(pos[1] - target[1])
    delta_z = abs(pos[2] - target[2])
    delta = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
    if not quite:
        print("Position:", pos)
        print("Delta:", delta)
        print("-------------------------")
    return delta


def move(x, y, z, *args, **kwargs):
    print("-------------------------")
    print("Move to:", x, y, z)
    print("-------------------------")
    if len(args) > 0:
        coords = [x, y, z, *args]
    else:
        coords = [x, y, z, *ANGLE["down"]]
    if "speed" in kwargs:
        speed = kwargs["speed"]
    else:
        speed = DEFAULT_SPEED
    mc.send_coords(coords, speed, MODE)
    while cmp(mc.get_coords(), (x, y, z)) > TARGET_DELTA:
        if cmp(mc.get_coords(), (x, y, z), quite=True) == 999:
            time.sleep(2)
            break
        time.sleep(0.1)
    time.sleep(0.5)
    return


def pump_on():
    mc.set_basic_output(2, 0)
    mc.set_basic_output(5, 0)
    time.sleep(2)
    return


def pump_off():
    mc.set_basic_output(2, 1)
    mc.set_basic_output(5, 1)
    time.sleep(4)
    return


def init():
    pump_off()
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(3)
    return


def pixel_to_coord(pixel_x, pixel_y):
    x = FAR_BORDER - pixel_y * (FAR_BORDER - NEAR_BORDER)
    y = LEFT_BORDER - pixel_x * (LEFT_BORDER - RIGHT_BORDER)
    return x, y


def grasp(pixel_x, pixel_y):
    init()
    x, y = pixel_to_coord(pixel_x, pixel_y)
    move(x, y, BOX_LEVEL + 30)
    move(x, y, BOX_LEVEL, speed=10)
    pump_on()
    move(x, y, BOX_LEVEL + 50)
    move(150, 0, 240)
    return


def put_off(label):
    move(*(POSITION[label][0]))
    move(*(POSITION[label][1]))
    pump_off()
    move(*(POSITION[label][0]))
    return


if __name__ == "__main__":
    init()
    grasp(0.5, 0.5)
    put_off("left-near")
