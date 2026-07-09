import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read Image
img = cv2.imread(r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\rgb_image.jpg')

# Convert BGR Image to RGB Image when use plt
# plt.imshow(img[:,:,::-1])

# Convert to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower red and upper red value
lower_red = np.array([20, 100, 100])
upper_red = np.array([30, 255, 255])

# mask1 cho vùng đỏ 1 (0 - 15)
mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
# print(type(mask1))
# print(mask1.dtype)
# print(mask1.shape)
# print(mask1.max())
# print(mask1.min())

lower_red2 = np.array([165, 120, 70])
upper_red2 = np.array([[179, 255, 255]])
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Gộp 2 mask 2 vùng lại => tìm vùng cần tìm và giữ lại (màu trắng)
final_mask = cv2.bitwise_or(mask1, mask2)
print(np.unique(final_mask))

# Hiển thị cái vùng được giữ lại ở mask lên ảnh gốc với màu gốc
res = cv2.bitwise_and(img, img, mask=mask1)

plt.figure(1)
plt.subplot(121)
plt.imshow(img[:,:,::-1])

plt.subplot(122)
plt.imshow(res[:,:,::-1])

plt.show()