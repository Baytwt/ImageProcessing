# 频域图像增强
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 原图
img = cv2.imread('./Pics/chrome.jpg',cv2.IMREAD_GRAYSCALE) # 灰度读取
plt.subplot(231),plt.imshow(img,'gray'),plt.title('original')

# 二维傅里叶变换,将图像由空间域变换到频域
fft2 = np.fft.fft2(img)
plt.subplot(232),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')

# 将图像变换的原点移动到频域矩阵的中心,并显示结果
shift2center = np.fft.fftshift(fft2)
plt.subplot(233),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')

# 对傅里叶变换的结果进行对数变换
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')

# 对中心化后的结果进行对数变换
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('shift2center')



plt.show()
