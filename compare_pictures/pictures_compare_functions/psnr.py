import numpy as np
import cv2
import math

  
def psnr(img1_path, img2_path):
    # img1_path and img2_path are picture path string
	#这里输入的是（0,255）的灰度或彩色图像，如果是彩色图像，则numpy.mean相当于对三个通道计算的结果再求均值
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10: # 如果两图片差距过小代表完美重合
        return 100
    PIXEL_MAX = 1.0

    psnr_result = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return round(psnr_result, 5)
