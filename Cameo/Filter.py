import cv2
import numpy as np
from scipy import ndimage   # 用来处理多维数组的卷积运算

kernel_3x3 = np.array([[-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread('1.png',0) # 读入灰度图

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

blurred = cv2.GaussianBlur(img, (11,11), 0)
g_hpf = img - blurred

# cv2.imshow("origin",img) # 原图
# cv2.imshow("3x3",k3)    # 3x3滤波
# cv2.imshow("5x5",k5)    # 5x5滤波
# cv2.imshow("g_hpf",g_hpf) # 高通滤波

''' 边缘检测 '''
cv2.imshow("canny",cv2.Canny(img,200,200))  # Canny边缘检测

''' 轮廓检测 '''
# 200x200黑色空白图像，在中间放置一个白色方块
# img2 = np.zeros((200, 200), dtype = np.uint8)
# img2[50:100, 50:150] = 255
#
# ret, thresh = cv2.threshold(img2, 127, 255, 0)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# img2 = cv2.drawContours(color, contours, -1, (0,255,0), 2)
# cv2.imshow("contours",color)

''' 边界框 '''
img3 = cv2.pyrDown(cv2.imread("chrome.jpg",cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img3.copy(),cv2.COLOR_BGR2GRAY), 127,255,cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    ''' 第一步：计算出一个简单的边界框 '''
    # find bounding box coorinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img3, (x,y), (x+w, y+h), (0, 255, 0), 2)

    ''' 第二步：计算出包围目标的最小矩形区域 '''
    # find minimum area
    rect = cv2. minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)
    # draw contours
    cv2.drawContours(img3, [box], 0, (0, 0, 255), 3)

    # calculate center and radius of minimum enclosing circle
    (x,y), radius = cv2. minEnclosingCircle(c)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img3 = cv2.circle(img3, center, radius, (0, 255, 0), 2)

cv2.drawContours(img3, contours, -1, (255, 0 ,0), 1)
cv2.imshow("contours",img3)

''' 凸轮廓 '''


cv2.waitKey()
cv2.destroyAllWindows()

#canny边缘检测
