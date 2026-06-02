import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# 隐藏 tkinter 主窗口
Tk().withdraw()

print("请选择第一张图片")
img1_path = askopenfilename(
    title="选择第一张图片",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
)

print("请选择第二张图片")
img2_path = askopenfilename(
    title="选择第二张图片",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
)

if not img1_path or not img2_path:
    print("未选择图片")
    exit()

# 兼容中文路径读取
img1 = cv2.imdecode(
    np.fromfile(img1_path, dtype=np.uint8),
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imdecode(
    np.fromfile(img2_path, dtype=np.uint8),
    cv2.IMREAD_GRAYSCALE
)

if img1 is None or img2 is None:
    print("图片读取失败")
    exit()

h1, w1 = img1.shape
h2, w2 = img2.shape

# 目标尺寸（较大的尺寸）
target_h = max(h1, h2)
target_w = max(w1, w2)

# 插值放大
if (h1, w1) != (target_h, target_w):
    img1 = cv2.resize(
        img1,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC
    )

if (h2, w2) != (target_h, target_w):
    img2 = cv2.resize(
        img2,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC
    )

# 计算绝对差分
diff = cv2.absdiff(img1, img2)

# 输出路径
base_name = os.path.splitext(os.path.basename(img1_path))[0]
ext = os.path.splitext(img1_path)[1]

output_path = os.path.join(
    os.path.dirname(img1_path),
    f"{base_name}_diff{ext}"
)

# 中文路径保存兼容
cv2.imencode(ext, diff)[1].tofile(output_path)

print(f"差分图已保存：{output_path}")