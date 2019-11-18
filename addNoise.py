import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# 加噪声
def sp_noise(image, prob):
    '''
    添加椒盐噪声
    :param prob:噪声比例
    '''
    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gauss_noise(image, mean=0, var=0.001):
    '''
    添加高斯噪声
    mean:均值
    var:方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def exp_noise(image, a=1):
    '''
    添加指数噪声
    a: a*exp(-a*x)
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.exponential(1/a,size=image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*a)
    return out

def alpha_mean(image):
    return image

img = cv2.imread('./Pics/noise_test.png',0)

# 噪声图像
img_salt=sp_noise(img,0.1)
img_gauss=gauss_noise(img)
img_exp=exp_noise(img,10)

# 处理噪声图像
# 算术均值滤波器5x5
img_salt_mean = cv2.blur(img_salt, (5,5))
img_gauss_mean = cv2.blur(img_gauss, (5,5))
img_exp_mean = cv2.blur(img_exp, (5,5))

# 阿尔法均值滤波器
# def alpha(img, kernel_size, d):
#     for i in img.shape[0]-kernel_size:
#         for j in img.shape[1]-kernel_size:


# 自适应中值滤波器


# 输出图像
titles = ['origin', 'salt','gauss','exponential',
          'salt_mean', 'gauss_mean', 'exponential_mean']
imgs = [img, img_salt, img_gauss, img_exp,
        img_salt_mean, img_gauss_mean, img_exp_mean]

for i in range(0,len(imgs)):
    plt.subplot(3,3,i+1)
    plt.imshow((imgs[i]),cmap=plt.cm.gray)  # 灰度
    plt.title(titles[i])
plt.show()
