# 画直方图

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Pics/windows.png')

cv2.namedWindow('input_image',cv2.WINDOW_NORMAL)
cv2.imshow('input_image',img)

# 画单通道图像的直方图
plt.hist(img.ravel(),256,[0,256])   #降为一位数组
plt.show()

# 画三通道图像的直方图
color = ('b','g','r')
for i, color in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])    #计算直方图
    plt.plot(hist,color)
    plt.xlim([0,256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

