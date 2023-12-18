from match import arrayToImage, imageTextMatch, classify
from proposal import capture, propose
from robot import init, grasp, put_off
from matplotlib import pyplot as plt


def vision_test(image):
    regions = propose(image)
    prompt = ["an apple", "a camera", "a gun"]
    for region in regions:
        img = arrayToImage(region["image"])
        probs = imageTextMatch(img, prompt)
        plt.imshow(img)
        print("Center:", "X =", region["x"], ",", "Y =", region["y"])
        print("Label probs:", probs)
        plt.show()
    return


if __name__ == '__main__':
    prompt = "an apple"
    print("Start")
    init()
    print("Capture")
    image = capture()
    print("Vision test")
    vision_test(image)
    print("Propose")
    regions = propose(image)
    print("Classify")
    images = list(map(lambda region: arrayToImage(region["image"]), regions))
    index = classify(prompt, images)
    grasp(regions[index]["x"], regions[index]["y"])
    put_off("left-near")
