from PIL import Image
import numpy as np
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name


def arrayToImage(array: np.ndarray):
    return Image.fromarray(array)
    # 假设 array 是一个形状为(3, H, W)的numpy数组
    # return Image.fromarray((array * 255).astype(np.uint8).transpose(1, 2, 0))


def imageTextMatch(image: Image.Image, text: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.eval()
    text = clip.tokenize(text).to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs

def TextMatchImages(text: str, images: list[Image.Image]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.eval()
    text = clip.tokenize([text]).to(device)
    images = torch.stack(list(map(preprocess, images))).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model.get_similarity(images, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
    return probs

def classify(prompt: str, images: list[Image.Image]):
    probs = TextMatchImages(prompt, images)
    print("Label probs: ", probs)
    ans = probs.argmax(axis=0)
    return ans
