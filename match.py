"""计算每张图片与文本串的匹配概率
    输入：
        images: 图片列表, list of PIL.Image.Image
        text: 文本串, str | str[]
    返回：
        probs: 每张图片的概率, (N, ), np.array
"""
from PIL import Image
import numpy as np
import torch
import clip

CONFIDENCE = 0.75


def arrayToImage(array: np.ndarray):
    return Image.fromarray(array)
    # 假设 array 是一个形状为(3, H, W)的numpy数组
    # return Image.fromarray((array * 255).astype(np.uint8).transpose(1, 2, 0))


def imageTextMatch(image: Image.Image, text: list[str]):
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(text).to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs


def classify(images: list[Image.Image], prompts: list[str]):
    ans = []
    for img in images:
        probs = imageTextMatch(img, prompts)
        index = np.where(probs > CONFIDENCE)[1]
        if len(index) > 0:
            ans.append(prompts[index[0]])
        else:
            ans.append("unknown")
    return ans
