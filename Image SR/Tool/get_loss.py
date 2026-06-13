import math
import numpy as np

PIXEL_MIN = 0.0
PIXEL_MAX = 255.0

def PSNR(pred, gt):
    imdff = pred - gt # get img diff
    rmse = math.sqrt(np.mean(imdff ** 2)) # float
    # RMSE = sqrt( (1/N) * Σ (pred_i - gt_i)² )
    if rmse == 0:
        return 100 # img same 100db
    return 20 * math.log10(PIXEL_MAX / rmse)
    # 255 img range