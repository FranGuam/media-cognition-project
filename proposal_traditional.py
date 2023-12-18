import numpy as np
import cv2
# import apriltag

CAMERA_ID = 0
GUASSIAN_KERNEL_SIZE = (3, 3)
CLOSE_KERNEL_SIZE = (3, 3)


def capture():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    cap.release()
    return frame


def apriltag(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    for result in results:
        # Get the tag ID and the corners
        tag_id = result.tag_id
        corners = result.corners

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

        # Print the tag ID and the position
        print(f"Tag ID: {tag_id}")
        print(f"Position: {tvec.ravel()}")
    return results


def crop(image):
    # TODO: Crop using apriltag
    x, y, w, h = 792, 536, 502, 506
    return image[y:y + h, x:x + w]


def prepare(image):
    # 对原图做高斯模糊
    image_guass = cv2.GaussianBlur(image, GUASSIAN_KERNEL_SIZE, 0)

    # 取灰度图
    image_gray = cv2.cvtColor(image_guass, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    image_canny = cv2.Canny(image_gray, 50, 150)
    # image_canny = cv2.Canny(image_gray, 100, 200)

    # 对边缘图进行开闭运算，腐蚀和膨胀
    kernel = np.ones(CLOSE_KERNEL_SIZE, np.uint8)
    image_canny_close = cv2.morphologyEx(image_canny, cv2.MORPH_CLOSE, kernel)
    # image_canny_open = cv2.morphologyEx(image_canny, cv2.MORPH_OPEN, kernel)
    # image_canny_erode = cv2.erode(image_canny, kernel, iterations=1)
    # image_canny_dilate = cv2.dilate(image_canny, kernel, iterations=1)

    # cv2.imshow("Gray", image_gray)
    # cv2.imshow("Canny", image_canny)
    # cv2.imshow("Open", image_canny_open)
    # cv2.imshow("Close", image_canny_close)
    # cv2.imshow("Erode", image_canny_erode)
    # cv2.imshow("Dilate", image_canny_dilate)

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
        # TODO: Update this value
        if _w > 50 and _h > 50:
            img = cv2.cvtColor(image[_y:_y + _h, _x:_x + _w], cv2.COLOR_BGR2RGB)
            ans.append({"x": x / w, "y": y / h, "image": img})
    return ans


def refine(regions):
    for region in regions:
        image = region["image"].copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        regs = propose(image)
        if len(regs) != 1:
            continue
        reg = regs[0]
        region["x"] = 2 * region["x"] * reg["x"] + (1 - 2 * reg["x"]) * region["corner_x"]
        region["y"] = 2 * region["y"] * reg["y"] + (1 - 2 * reg["y"]) * region["corner_y"]
        region["image"] = reg["image"]
    return regions


if __name__ == '__main__':
    image = capture()
    _, _, image = crop(image)
    cv2.imshow("Original", image)
    canny = prepare(image)
    cv2.imshow("Canny", canny)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 0, 255), 1)
    cv2.imshow("Contours", image_contours)

    for contour in contours:
        x, y = center(contour)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Result", image)

    # # 以以上结果作为掩码图，对原图进行操作，获取全部的封闭区域
    # mask = image_canny_close.copy()
    # mask[mask > 0] = 255
    # result_img = cv2.bitwise_and(image, image, mask=mask)

    if cv2.waitKey() == ord('q'):
        cv2.destroyAllWindows()
