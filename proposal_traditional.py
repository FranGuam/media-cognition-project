import numpy as np
import cv2
import pupil_apriltags as apriltag  # for windows
# import apriltag  # for linux

CAMERA_ID = 0
TAG_LEFT_NEAR = 0
TAG_RIGHT_FAR = 0
TAG_LEFT_FAR = 2
TAG_RIGHT_NEAR = 2
GUASSIAN_KERNEL_SIZE = (3, 3)
CLOSE_KERNEL_SIZE = (3, 3)
BOX_MIN_WIDTH = 120
BOX_MIN_HEIGHT = 120


def show(caption, image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(caption, img)
    

def capture():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print("Adjusting resolution")
    # This is time-consuming
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Capture image")
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    print("Capture success")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame


def detect(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))  # for linux
    detector = apriltag.Detector(families="tag16h5")  # for windows
    tags = detector.detect(image_gray)
    result = {}
    # image_detect = image.copy()
    for tag in tags:
        # cv2.polylines(image_detect, [np.array(tag.corners, np.int32)], True, (0, 255, 0), 2)
        # cv2.putText(image_detect, str(tag.tag_id), np.array(tag.corners[0], np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if tag.tag_id == TAG_LEFT_NEAR:
            result["x_min"] = round(min(tag.corners[:, 0]))
            result["y_max"] = round(max(tag.corners[:, 1]))
        elif tag.tag_id == TAG_RIGHT_FAR:
            result["x_max"] = round(max(tag.corners[:, 0]))
            result["y_min"] = round(min(tag.corners[:, 1]))
        elif tag.tag_id == TAG_LEFT_FAR:
            result["x_min"] = round(min(tag.corners[:, 0]))
            result["y_min"] = round(min(tag.corners[:, 1]))
        elif tag.tag_id == TAG_RIGHT_NEAR:
            result["x_max"] = round(max(tag.corners[:, 0]))
            result["y_max"] = round(max(tag.corners[:, 1]))
        else:
            print("Unknown tag ID:", tag.tag_id)
            print("Tag corners:", tag.corners)
    # show("Detect", image_detect)
    if "x_min" not in result:
        result["x_min"] = 792
    if "x_max" not in result:
        result["x_max"] = 1294
    if "y_min" not in result:
        result["y_min"] = 536
    if "y_max" not in result:
        result["y_max"] = 1042
    return result


def crop(image):
    result = detect(image)
    return image[result["y_min"]:result["y_max"], result["x_min"]:result["x_max"]]


def motion(image):
    corners = detect(image)
    # Define the tag size (in meters)
    tag_size = 0.1

    # Define the 3D coordinates of the tag corners in the world frame
    obj_pts = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, tag_size / 2, 0],
        [-tag_size / 2, tag_size / 2, 0]
    ])

    # Reshape the 2D coordinates of the tag corners in the image frame
    img_pts = corners.reshape(4, 2)

    # Define the camera matrix (fx, fy, cx, cy) and the distortion coefficients
    # You can obtain these values by calibrating your camera
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([0, 0, 0, 0])

    # Solve for the pose of the tag in the camera frame
    _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
    return rvec, tvec


def prepare(image):
    # 对原图做高斯模糊
    image_guass = cv2.GaussianBlur(image, GUASSIAN_KERNEL_SIZE, 0)

    # 取灰度图
    image_gray = cv2.cvtColor(image_guass, cv2.COLOR_RGB2GRAY)

    # Canny边缘检测
    image_canny = cv2.Canny(image_gray, 50, 150)
    # image_canny = cv2.Canny(image_gray, 100, 200)

    # 对边缘图进行开闭运算，腐蚀和膨胀
    kernel = np.ones(CLOSE_KERNEL_SIZE, np.uint8)
    image_canny_close = cv2.morphologyEx(image_canny, cv2.MORPH_CLOSE, kernel)
    # image_canny_open = cv2.morphologyEx(image_canny, cv2.MORPH_OPEN, kernel)
    # image_canny_erode = cv2.erode(image_canny, kernel, iterations=1)
    # image_canny_dilate = cv2.dilate(image_canny, kernel, iterations=1)

    # show("Guass", image_guass)
    # show("Gray", image_gray)
    # show("Canny", image_canny)
    # show("Close", image_canny_close)

    return image_canny_close


def center(contour):
    # 计算轮廓的几何矩
    M = cv2.moments(contour)

    # 判断分母是否为零，避免出现错误
    if M["m00"] != 0:
        # 根据公式计算重心的坐标
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        # 如果分母为零，就取轮廓的第一个点作为重心
        x = contour[0][0][0]
        y = contour[0][0][1]
    return x, y


def propose(image):
    h, w, _ = image.shape
    canny = prepare(image)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ans = []
    for contour in contours:
        x, y = center(contour)
        _x, _y, _w, _h = cv2.boundingRect(contour)
        if _w > BOX_MIN_WIDTH and _h > BOX_MIN_HEIGHT:
            ans.append({"x": x / w, "y": y / h, "image": image[_y:_y + _h, _x:_x + _w]})
    return ans


def refine(regions):
    for i, region in enumerate(regions):
        regs = propose(region["image"])
        if len(regs) != 1:
            print("Refine failed for region", i, ". Detected", len(regs), "regions.")
            continue
        reg = regs[0]
        region["x"] = 2 * region["x"] * reg["x"] + (1 - 2 * reg["x"]) * region["corner_x"]
        region["y"] = 2 * region["y"] * reg["y"] + (1 - 2 * reg["y"]) * region["corner_y"]
        region["image"] = reg["image"]
    return regions


if __name__ == '__main__':
    image = capture()
    show("Original", image)
    image = crop(image)
    show("Crop", image)
    canny = prepare(image)
    show("Canny", canny)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 0, 255), 1)
    show("Contours", image_contours)

    for contour in contours:
        x, y = center(contour)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        x, y, w, h = cv2.boundingRect(contour)
        if w > BOX_MIN_WIDTH and h > BOX_MIN_HEIGHT:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show("Result", image)

    # # 以以上结果作为掩码图，对原图进行操作，获取全部的封闭区域
    # mask = image_canny_close.copy()
    # mask[mask > 0] = 255
    # result_img = cv2.bitwise_and(image, image, mask=mask)

    if cv2.waitKey() == ord('q'):
        cv2.destroyAllWindows()
