import numpy as np
import cv2

  
def mse_grey(img1_path, img2_path):
    # img1_path and img2_path are picture path string
	# the 'Mean Squared error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mse_grey_result = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    mse_grey_result /= float(img1.shape[0] * img2.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return round(mse_grey_result, 3)
