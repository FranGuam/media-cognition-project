from match import arrayToImage, imageTextMatch, classify
from proposal import capture, propose
from robot import init, grasp, put_off
from matplotlib import pyplot as plt

PROMPT_SET = [
    "an apple",
    "a camera",
    "a gun",
    "a ball",
    "a cat",
]


def vision_test():
    image = capture()
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
    init()
    image = capture()
    regions = propose(image)
    # vision_test()
    img = list(map(lambda region: arrayToImage(region["image"]), regions))
    category = classify(img, PROMPT_SET)
    print(category)
    for i in range(3):
        prompt = PROMPT_SET[i]
        index = category.index(prompt)
        grasp(regions[index]["x"], regions[index]["y"])
        put_off("left-near")
