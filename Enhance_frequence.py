# 频域图像增强
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 原图
img = cv2.imread('./Pics/apple.jpg',cv2.IMREAD_GRAYSCALE) # 灰度读取

# 二维傅里叶变换,将图像由空间域变换到频域
fft2 = np.fft.fft2(img)

# 将图像变换的原点移动到频域矩阵的中心
fshift = np.fft.fftshift(fft2)

# 低通滤波器

# 在频域取d将其反变换到空间域
def make_transform_matrix(d,image):
    ''' 低通滤波器
    d:半径
    '''
    transfor_matrix = np.zeros(image.shape)
    center_point = tuple(map(lambda x:(x-1)/2, fshift.shape))
    for i in range(transfor_matrix.shape[0]):
        for j in range(transfor_matrix.shape[1]):
            def cal_distance(pa,pb):
                from math import sqrt
                dis = sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                return dis
            dis = cal_distance(center_point, (i,j))
            if dis <= d:
                transfor_matrix[i,j] = 1
            else:
                transfor_matrix[i,j] = 0
    return transfor_matrix

d_1 = make_transform_matrix(10,fshift)
d_2 = make_transform_matrix(20,fshift)
d_3 = make_transform_matrix(50,fshift)

# 低通滤波
img_d1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_1)))
img_d2 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_2)))
img_d3 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_3)))

plt.subplot(231),plt.imshow(img,'gray'),plt.title('origin')
plt.subplot(234),plt.imshow(img_d1,'gray'),plt.title('d=10')
plt.subplot(235),plt.imshow(img_d2,'gray'),plt.title('d=20')
plt.subplot(236),plt.imshow(img_d3,'gray'),plt.title('d=50')

# 对傅里叶变换的结果进行对数变换 目的是将数据变化到0~255
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(232),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')
# 对中心化后的结果进行对数变换
log_shift2center = np.log(1 + np.abs(fshift))
plt.subplot(233),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')


plt.show()
