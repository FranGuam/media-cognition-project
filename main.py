import keyboard
from matplotlib import pyplot as plt

from audio import recognize
from proposal_traditional import crop, capture, propose, refine
from proposal_yolo import *
from match import arrayToImage, imageTextMatch, classify, openFromFile
from robot import init, grasp, put_off

APPROACH = "yolo+traditional"
DEFAULT_PROMPT = "Put an apple into the green bin"
BIN_IMAGES = [
    "./image/gray.jpg",
    "./image/green.jpg",
    "./image/red.jpg",
    "./image/blue.jpg"
]
BIN = ["left-near", "left-far", "right-near", "right-far"]


def vision_test():
    image = capture()
    regions = propose(image)
    PROMPT_SET = ["an apple", "a camera", "a gun", "a flower", "a cat", "a dog", "a shoe", "a clock"]
    for region in regions:
        img = arrayToImage(region["image"])
        probs = imageTextMatch(img, PROMPT_SET)
        plt.imshow(img)
        print("Center:", "X =", region["x"], ",", "Y =", region["y"])
        print("Label probs:", probs)
        plt.show()
    return


def main():
    print("==================== Voice to Text ====================")
    prompt = recognize()
    if prompt is None:
        prompt = DEFAULT_PROMPT
    print("Prompt:", prompt)
    prompt = "a shoe, green bin"
    print("==================== Capture Photo ====================")
    image = capture()
    arrayToImage(image).show("Original")
    print("==================== Propose Regions ====================")
    if APPROACH == "traditional":
        image = crop(image)
        plt.imshow(arrayToImage(image))
        plt.show()
        regions = propose(image)
    elif APPROACH == "yolo":
        model = Detr(lr=2.5e-6, weight_decay=1e-5)
        model.load_state_dict(torch.load('parameters.pth'))
        regions = yolos_proposal(model, image)
    elif APPROACH == "yolo+traditional":
        model = Detr(lr=2.5e-6, weight_decay=1e-5)
        model.load_state_dict(torch.load('parameters.pth'))
        regions = yolos_proposal(model, image)
        print("==================== Refine Regions ====================")
        regions = refine(regions)
    else:
        raise NotImplementedError
    for region in regions:
        plt.imshow(arrayToImage(region["image"]))
        plt.show()
    print("==================== Judge for object ====================")
    images = list(map(lambda region: arrayToImage(region["image"]), regions))
    object_index = classify(prompt, images)
    print("Index:", object_index)
    print("Center:", "X =", regions[object_index]["x"], ",", "Y =", regions[object_index]["y"])
    print("==================== Classify for bins ====================")
    bins = openFromFile(BIN_IMAGES)
    bin_index = classify(prompt, bins)
    print("Bin:", BIN[bin_index], "(", BIN_IMAGES[bin_index], ")")
    print("==================== Grasp and Put ====================")
    grasp(regions[object_index]["x"], regions[object_index]["y"])
    put_off(BIN[bin_index])
    init()
    print("Press Enter to restart / Press Esc to exit ...")


if __name__ == '__main__':
    print("==================== Initializing ====================")
    init()
    print("Press Enter to start ...")
    keyboard.add_hotkey('enter', main)
    keyboard.wait('esc')
