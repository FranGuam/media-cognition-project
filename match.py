from PIL import Image
import numpy as np
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

CONFIDENCE = 0.5


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


def classify(images: list[Image.Image], prompts: list[str]):
    ans = []
    for img in images:
        probs = imageTextMatch(img, prompts)
        print('probs:', probs)
        index = np.where(probs > CONFIDENCE)[1]
        if len(index) > 0:
            ans.append(prompts[index[0]])
        else:
            ans.append("unknown")
    return ans
