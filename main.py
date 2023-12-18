from match import arrayToImage, imageTextMatch, classify, openFromFile
from proposal import capture, propose
from robot import init, grasp, put_off
from audio import recognize
from yolo_proposal import *
from matplotlib import pyplot as plt

APPROACH = "yolo"
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
    if APPROACH == "traditional":
        regions = propose(image)
    elif APPROACH == "yolo":
        model = Detr(lr=2.5e-6, weight_decay=1e-5)
        model.load_state_dict(torch.load('parameters.pth'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        regions = yolos_proposal(model, image)
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
