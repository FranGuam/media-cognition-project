from match import arrayToImage, imageTextMatch, classify
from proposal import capture, propose
from robot import init, grasp, put_off
from audio import recognize
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    print("==================== Start ====================")
    init()
    print("==================== Voice to Text ====================")
    prompt = recognize()
    print("Prompt:", prompt)
    print("==================== Capture Photo ====================")
    image = capture()
    plt.imshow(arrayToImage(image))
    plt.show()
    print("==================== Propose Regions ====================")
    regions = propose(image)
    for region in regions:
        plt.imshow(arrayToImage(region["image"]))
        plt.show()
    print("==================== Classify ====================")
    images = list(map(lambda region: arrayToImage(region["image"]), regions))
    index = classify(prompt, images)
    print("Index:", index)
    print("Center:", "X =", regions[index]["x"], ",", "Y =", regions[index]["y"])
    print("==================== Grasp and Put ====================")
    grasp(regions[index]["x"], regions[index]["y"])
    put_off("left-near")
    init()
