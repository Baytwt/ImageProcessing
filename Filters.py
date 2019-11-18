# 中值滤波,用opencv实现

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from PIL import Image

img = cv2.imread('./Pics/leena.jpg')

'''
低通滤波器:消除噪声
高通滤波器:提取边缘
核:一个矩阵
'''

# 均值滤波
img_mean = cv2.blur(img, (5,5))

# 高斯滤波
img_Gaussian = cv2.GaussianBlur(img, (5,5), 0)

# 中值滤波
img_median = cv2.medianBlur(img, 5)

# 双边滤波
img_bilater = cv2.bilateralFilter(img, 9, 75, 75)



titles = ['origin', 'mean', 'Gaussian', 'median', 'bilateral']
imgs = [img, img_mean, img_Gaussian, img_median, img_bilater]

# 将通道顺序调换,方便plt输出
def cv2plt(image):
    b,g,r=cv2.split(image)
    return cv2.merge([r,g,b])

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(cv2plt(imgs[i]))
    plt.title(titles[i])
plt.show()