from match import arrayToImage, imageTextMatch, classify
from proposal import capture, propose,crop
# from robot import init, pixel_to_coord, grasp, put_off
from matplotlib import pyplot as plt
from yolo_proposal import *

prompt = "摄像机"


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
    # init()
    model = Detr(lr=2.5e-6, weight_decay=1e-5)
    model.load_state_dict(torch.load('parameters.pth'))         # Read the parameters prepared already
    # image = capture()
    image = cv2.imread("D:\code\media_cognitionProject\WIN_20231218_20_31_36_Pro.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # h,w,image = crop(image)
    # cv2.imshow("Original", image)
    regions = yolos_proposal(model,image)
    # vision_test()
    # print(regions)
    img = list(map(lambda region: arrayToImage(region["image"]), regions))
    # print(img)
    for image in img:
        plt.imshow(image)
        plt.show()
    index = classify(prompt, img)
    print(index)
