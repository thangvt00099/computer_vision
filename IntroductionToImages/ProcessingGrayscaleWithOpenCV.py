import numpy as np
import cv2

path = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein.webp'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(type(img))
print(img.dtype)
print(img.shape)

img[200:270, 280:350] = 255
cv2.imshow("Gray", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
