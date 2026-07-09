import numpy as np
import matplotlib.pyplot as plt
import cv2

path = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\rgb_image.jpg'
img = cv2.imread(path)
print(img.dtype)
print(img.shape)
print(type(img))

img = img[:,:,::-1]
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

R[100:400, 100:400] = 255
G[100:400, 100:400] = 0
B[100:400, 100:400] = 0
img[:, :, 0] = R
img[:, :, 1] = G
img[:, :, 2] = B

# plt.figure() - khung chứa toàn bộ biểu đồ
plt.figure(1)

# plt.subplot (nrows, ncols, index) - Tạo nhiều biểu đổ 1 ô
plt.subplot(231)
plt.imshow(img)

# plt.xticks (yticks) - loại bỏ trục x và y
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

# Red channel
plt.subplot(234)
plt.imshow(R, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Red Channel")

# Green channel
plt.subplot(235)
plt.imshow(G, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Green Channel")

# Blue channel
plt.subplot(236)
plt.imshow(B, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Blue Channel")

rgb = np.zeros((100, 150, 3), dtype='uint8')
rgb[:, 0:50, 0] = 255
rgb[:, 50:100, 1] = 255
rgb[:, 100:150, 2] = 255
plt.imshow(rgb)
plt.show()