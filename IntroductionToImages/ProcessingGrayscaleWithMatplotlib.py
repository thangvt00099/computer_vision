import numpy as np
import matplotlib.pyplot as plt

img = plt.imread(r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg')
print(type(img))
print(img.shape)
print(img.dtype)

img2 = img.copy()

img2[200:270, 280:350] = 255
img2[200:270, 200:270] = 255
plt.imshow(img2, cmap='gray')
plt.show()
# plt.imsave(r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg', img2, cmap='gray')