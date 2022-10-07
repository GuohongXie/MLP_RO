import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_offical


def ssim_multi_channel(img1_path, img2_path):
    ##### img1_path and img2_path are picture path string

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_multi_channel_result = ssim_offical(img1, img2, multichannel=True)

    return round(ssim_multi_channel_result, 6)
