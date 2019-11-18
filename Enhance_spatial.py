# 空间域图像增强
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Pics/lena_std.tif')

# 均值滤波
# img_mean = cv2.blur(img, (5,5))

# 高斯滤波
# img_Gaussian = cv2.GaussianBlur(img, (5,5), 0)

# 双边滤波
# img_bilater = cv2.bilateralFilter(img, 9, 75, 75)

# 中值滤波
img_median = cv2.medianBlur(img, 5)
cv2.imwrite('./Pics/lena_std_median.tif',img_median)
# 拉布拉斯变换
img_Laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=5)
cv2.imwrite('./Pics/lena_std_Laplacian.tif',img_Laplacian)

titles = ['origin', 'median', 'Laplacian']
imgs = [img, img_median, img_Laplacian]

# 将通道顺序调换,方便plt输出
def cv2plt(image):
    b,g,r=cv2.split(image)
    return cv2.merge([r,g,b])

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(cv2plt(imgs[i]))
    plt.title(titles[i])
plt.show()