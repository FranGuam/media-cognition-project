from pymycobot.mycobot import MyCobot
import time

BOX_LEVEL = 105
LEFT_BORDER = 75
RIGHT_BORDER = -35
NEAR_BORDER = 135
FAR_BORDER = 215
LEFT_PIXEL = 0
RIGHT_PIXEL = 256
NEAR_PIXEL = 256
FAR_PIXEL = 0

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
         [120, -150, 175, -180, 0, 0]],  # down
    "right-far":
        [[200, -100, 235, -180, 0, 0],  # up
         [220, -140, 220, -150, -10, -80]],  # down
}

ANGLE = {
    "down": [180, 0, 180],
}

mc = MyCobot('COM3', 115200)


def cmp(pos, target):
    if len(pos) == 0:
        return 999
    delta_x = abs(pos[0] - target[0])
    delta_y = abs(pos[1] - target[1])
    delta_z = abs(pos[2] - target[2])
    delta = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
    print("Position:", pos)
    print("Delta:", delta)
    print("-------------------------")
    return delta


def move(x, y, z, *args, **kwargs):
    if len(args) > 0:
        coords = [x, y, z, *args]
    else:
        coords = [x, y, z, *ANGLE["down"]]
    if "speed" in kwargs:
        speed = kwargs["speed"]
    else:
        speed = DEFAULT_SPEED
    mc.send_coords(coords, speed, MODE)
    start = time.time()
    while cmp(mc.get_coords(), (x, y, z)) > TARGET_DELTA:
        if cmp(mc.get_coords(), (x, y, z)) == 999:
            time.sleep(2)
            break
        assert time.time() - start < 3
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
    x = FAR_BORDER - (pixel_x - FAR_PIXEL) / (NEAR_PIXEL - FAR_PIXEL) * (FAR_BORDER - NEAR_BORDER)
    y = LEFT_BORDER - (pixel_y - LEFT_PIXEL) / (RIGHT_PIXEL - LEFT_PIXEL) * (LEFT_BORDER - RIGHT_BORDER)
    return x, y


def grasp(pos):
    init()
    move(pos[0], pos[1], BOX_LEVEL + 30)
    move(pos[0], pos[1], BOX_LEVEL, speed=10)
    pump_on()
    move(pos[0], pos[1], BOX_LEVEL + 150)
    return


def put_off(label):
    move(*(POSITION[label][0]))
    move(*(POSITION[label][1]))
    pump_off()
    move(*(POSITION[label][0]))
    return


if __name__ == "__main__":
    init()
    # Take a picture
    grasp(pixel_to_coord(128, 128))
    put_off("left-near")
