# media-cognition-project

Class project for Media and Cognition 2023 Fall

### 机械臂相关数据

`coords = [x, y, z, rx, ry, rz]`

面向屏幕
- x：前后，远离屏幕为正，最大为284，最小为-284 
- y：左右，左为正，最大为290，最小为-290
- z：上下，上为正，最大为410，桌面为70 
- rx：yz面，上为0，屏幕前往后看顺时针
- ry：xz面，下为0，从左往右看逆时针
- rz：xy面，远离屏幕为0，从上向下看逆时针

### CLIP

接口函数 `ImageTextMatch`，计算传入的每张图片与文本串的匹配概率。使用例见 `main.py`。

- 环境配置

  [openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image (github.com)](https://github.com/openai/CLIP#modelencode_texttext-tensor)

  ```
  conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  ```

### Chinese-Clip

[OFA-Sys/Chinese-CLIP: Chinese version of CLIP which achieves Chinese cross-modal retrieval and representation generation. (github.com)](https://github.com/OFA-Sys/Chinese-CLIP)

- 环境配置

  ```
  python >= 3.6.4
  pytorch >= 1.8.0 (with torchvision >= 0.9.0)
  CUDA Version >= 10.2
  ```

  ```
  numpy
  tqdm
  six
  timm
  patch_ng
  lmdb==1.3.0
  torch>=1.7.1
  torchvision
  
  cn_clip
  ```

  
