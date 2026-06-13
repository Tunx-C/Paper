import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.merge import merge

def merge_patches_to_png(tif_paths, output_png, lower_percent=2, upper_percent=98):
    """
    将同一时间步的所有 patch tif 拼接成完整图像，并保存为 RGB PNG。
    红色=VV, 绿色=VH, 蓝色=0
    """
    # 打开所有 tif 文件（不立即读入内存）
    src_files = [rasterio.open(p) for p in tif_paths]

    # 拼接（自动根据地理坐标布局）
    mosaic, out_transform = merge(src_files)
    # mosaic 形状: (band_count, height, width)

    # 关闭打开的文件
    for src in src_files:
        src.close()

    if mosaic.shape[0] < 2:
        raise ValueError("需要至少两个波段 (VV 和 VH)")

    vv = mosaic[0].astype(np.float32)
    vh = mosaic[1].astype(np.float32)

    # ----- 转 dB 并消除无效值 -----
    def safe_to_db(arr):
        arr = arr.copy()
        arr[arr <= 0] = np.nan
        with np.errstate(invalid='ignore'):
            return 10 * np.log10(arr)

    vv_db = safe_to_db(vv)
    vh_db = safe_to_db(vh)

    # ----- 百分比拉伸 -----
    def percent_stretch(img, low, high):
        p_low = np.nanpercentile(img, low)
        p_high = np.nanpercentile(img, high)
        if np.isnan(p_low) or np.isnan(p_high):
            return np.zeros_like(img, dtype=np.uint8)
        clipped = np.clip(img, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low + 1e-8) * 255.0
        return np.nan_to_num(norm, nan=0).astype(np.uint8)

    vv_8u = percent_stretch(vv_db, lower_percent, upper_percent)
    vh_8u = percent_stretch(vh_db, lower_percent, upper_percent)

    # ----- 合成 RGB -----
    h, w = vv_8u.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = vv_8u
    rgb[:, :, 1] = vh_8u

    Image.fromarray(rgb).save(output_png)
    print(f"已保存完整拼接 PNG: {output_png}")


if __name__ == "__main__":
    # ========== 修改为你的 ROI 基础路径 ==========
    roi_dir = Path(r"T:\Proj\Data\SEN12MS-CR-TS\test\s1\ROIs1868\73")

    # 遍历所有时间步文件夹 (0,1,2...)
    for step_dir in sorted(roi_dir.iterdir()):
        if not step_dir.is_dir():
            continue

        # 找到该时间步下所有 .tif 文件（即所有 patch）
        tif_list = sorted(step_dir.glob("*.tif"))
        if not tif_list:
            print(f"跳过空目录: {step_dir}")
            continue

        # 输出 PNG 放在时间步文件夹内，命名为 merged_rgb.png
        out_png = step_dir / "merged_rgb.png"

        # 拼接并转换
        try:
            merge_patches_to_png(tif_list, str(out_png))
        except Exception as e:
            print(f"处理 {step_dir} 时出错: {e}")