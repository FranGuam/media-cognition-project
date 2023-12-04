from PIL import Image
import numpy as np
from ImageTextMatch import ImageTextMatch

image = Image.open("CLIP.png")
probs = ImageTextMatch([image], ["a diagram", "a dog", "a cat"])
print("Label probs:", probs)

"""
# 假设 img_np 是一个形状为(3, H, W)的numpy数组
img_np = np.random.rand(3, 224, 224)

# 将numpy数组转换为PIL图像对象
img_pil = Image.fromarray((img_np * 255).astype(np.uint8).transpose(1, 2, 0))
"""