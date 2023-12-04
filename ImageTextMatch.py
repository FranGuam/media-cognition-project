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

def ImageTextMatch(images: list[Image.Image], text: str | list[str]):
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(text).to(device)
    for image in images:
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs

