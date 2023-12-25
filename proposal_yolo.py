import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import numpy
import torch
import pupil_apriltags as apriltag

MIN_X = 770
MIN_Y = 540


class Detr(pl.LightningModule):

     def __init__(self, lr, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
                                                             num_labels=64,
                                                             ignore_mismatched_sizes=True)
         self.config = DetrConfig()
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.weight_decay = weight_decay

     def forward(self, pixel_values):
       outputs = self.model(pixel_values=pixel_values)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, labels=labels)
       
       loss = outputs.loss
       loss_dict = outputs.loss_dict

       #print(loss,loss_dict)
       return loss, loss_dict
     


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（Intersection over Union）。

    参数：
        - box1: 第一个边界框，形状为 [4] 的张量，包含左上角和右下角的坐标（x1, y1, x2, y2）。
        - box2: 第二个边界框，形状为 [4] 的张量，包含左上角和右下角的坐标（x1, y1, x2, y2）。

    返回值：
        - iou: 交并比（IoU）值，标量值（float）。

    注意：这里假设输入的边界框张量使用了左上角和右下角的表示方式。
    """

    # 提取边界框坐标
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # 计算相交部分的坐标
    x1_intersection = torch.max(x1_box1, x1_box2)
    y1_intersection = torch.max(y1_box1, y1_box2)
    x2_intersection = torch.min(x2_box1, x2_box2)
    y2_intersection = torch.min(y2_box1, y2_box2)

    # 计算相交部分的宽度和高度
    width_intersection = torch.clamp(x2_intersection - x1_intersection, min=0)
    height_intersection = torch.clamp(y2_intersection - y1_intersection, min=0)

    # 计算相交部分的面积
    area_intersection = width_intersection * height_intersection

    # 计算并集的面积
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    area_union = area_box1 + area_box2 - area_intersection
    area_union = min(area_box1, area_box2)
    # 计算交并比（IoU）
    iou = area_intersection / area_union

    return iou.item()

def new_box(box1, box2):
    ''' 
    合并两个边界框,返回并集
    '''
    # 提取边界框坐标
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # 计算相交部分的坐标
    x1_intersection = torch.min(x1_box1, x1_box2)
    y1_intersection = torch.min(y1_box1, y1_box2)
    x2_intersection = torch.max(x2_box1, x2_box2)
    y2_intersection = torch.max(y2_box1, y2_box2)

    return torch.FloatTensor([x1_intersection, y1_intersection, x2_intersection, y2_intersection])



def detect(image):
    '''
    return minx, miny, height, width
    '''
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))  # for linux
    detector = apriltag.Detector(families="tag16h5")  # for windows
    tags = detector.detect(image_gray)
    if(len(tags) < 2):
        # if not detected, return default value
        return MIN_X,MIN_Y,550, image.size[1] - MIN_Y
    corners = numpy.array([tag.corners for tag in tags])
    min_x = min(corners[:,0])
    max_x = max(corners[:,0])
    min_y = min(corners[:,1])
    max_y = max(corners[:,1])
    return min_x,min_y,max_x-min_x,max_y-min_y


from PIL import Image, ImageDraw, ImageFont
    
def yolos_proposal(model, ori_image : numpy.ndarray):

    min_x, min_y, width, height = MIN_X,MIN_Y,550, 520
    image = Image.fromarray(ori_image)


#    image = Image.open("D:\code\media_cognitionProject\WIN_20231218_16_57_45_Pro.jpg")

    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    #model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    device = torch.device('cpu')
    model.to(device)

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # get results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
    
    # boxes的数据格式是: [n, x1, y1, x2, y2]
    boxes = results["boxes"]

    # 合并边界框
    i = 0
    j = 0
    while i < len(boxes):
        j = i + 1
        while j <len(boxes):
            print(i,j,calculate_iou(boxes[i],boxes[j]))
            if calculate_iou(boxes[i],boxes[j]) > 0.4:
                boxes[i] = new_box(boxes[i],boxes[j])
                # 要删除的行索引
                row_index = j

                # 使用 torch.cat 删除这一行
                boxes = torch.cat((boxes[:row_index], boxes[row_index+1:]))
                j = j - 1
            j = j + 1
        i = i + 1

    # 绘制边界框
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    font_size = 16
    # font = ImageFont.load_default().font_variant(size=font_size)

    for  box in boxes:
        # 取整并转换为整数
        box = [round(i, 2) for i in box.tolist()]
        box = [int(i) for i in box]
        
        # 绘制边界框矩形
        draw.rectangle(box, outline="red",width=4)


    image_copy.show()
    
    # 裁剪出propose的区域， 返回值是一个list，每个元素是一个字典，包含x,y,image
    regions = []
    # width = 550
    # print(image.size)
    # height = image.size[1] - MIN_Y
    for box in boxes:
        # 取整并转换为整数
        box = [round(i, 2) for i in box.tolist()]
        box = [int(i) for i in box]
        center_x = (box[0] + box[2]) / 2 - min_x
        center_x = int(center_x)
        center_y = (box[1] + box[3]) / 2 - min_y
        center_y = int(center_y)

        if 1:
            img = image.crop(box)
            img_cv = numpy.array(img)
            # cv2.imshow("crop", img_cv)
            print(center_x/width,center_y/height)
            regions.append({
                "x": center_x/width,
                "y": center_y/height,
                "corner_x": (box[0] - min_x) / width,
                "corner_y": (box[1] - min_y) / height,
                "image": img_cv
            })
            print(center_x/width,center_y/height)
    
    return regions

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = Detr(lr=2.5e-6, weight_decay=1e-5)
    model.load_state_dict(torch.load('parameters.pth'))         # Read the parameters prepared already

    image = cv2.imread("E:\Resources\media-cognition-project\image\whole.jpg")
    #h,w,image = crop(image)
    plt.figure()
    plt.imshow(image)
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(image[MIN_Y: MIN_Y + 520,MIN_X:MIN_X + 550])
    plt.waitforbuttonpress()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    regions = yolos_proposal(model,image)
    for region in regions:
        print(region["x"],region["y"])
        img = region["image"]
        plt.figure()
        plt.imshow(img)
        plt.waitforbuttonpress()
