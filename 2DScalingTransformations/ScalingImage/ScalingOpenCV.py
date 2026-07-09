import numpy as np
import matplotlib.pyplot as plt
import cv2

grayImage = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg'
colorImage = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\rgb_image.jpg'

I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)
I_BGR = cv2.imread(colorImage)
print(I_gray.shape)

plt.imshow(I_gray, cmap='gray')
plt.imshow(I_BGR[:,:,::-1])

I_gray_resized = cv2.resize(src=I_gray, fx=1.5, fy=0.8, dsize=None)
print(I_gray_resized.shape)
plt.imshow(I_gray_resized, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()