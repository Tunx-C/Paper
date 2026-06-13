import os
from PIL import Image

def downsample_images(gt_dir, scale=4):
    # LR目录
    lr_dir = os.path.join(os.path.dirname(gt_dir), "LR")
    os.makedirs(lr_dir, exist_ok=True)

    # 支持的图片格式
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for root, _, files in os.walk(gt_dir):
        for file in files:
            if file.lower().endswith(exts):
                gt_path = os.path.join(root, file)

                # 打开图片
                img = Image.open(gt_path).convert("RGB")

                # 计算尺寸（4x下采样）
                w, h = img.size
                lr_size = (w // scale, h // scale)

                # Bicubic下采样
                img_lr = img.resize(lr_size, Image.BICUBIC)

                # 保存路径（只保留文件名，不保留子目录结构）
                save_path = os.path.join(lr_dir, file)

                img_lr.save(save_path)
                print(f"Saved: {save_path}")

if __name__ == "__main__":
    gt_dir = r"../Image SR/2024 FreMamba RSI SR/codes/dataload/DIOR/GT"  # 改成你的GT目录
    downsample_images(gt_dir, scale=4)