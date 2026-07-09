import cv2
import matplotlib.pyplot as plt
import numpy as np

grayImage = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg'
I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)

numRows = I_gray.shape[0]
numCols = I_gray.shape[1]

print(f"numRows: {numRows}, numCols: {numCols}")

I_gray_copy = np.zeros((numRows, numCols), dtype='uint8')

# Copy original image
# for i in range(numRows):
#     for j in range(numCols):
#         I_gray_copy[i, j] = I_gray[i, j]

# Reverse Image by vertical
# for i in range(numRows):
#     for j in range(numCols):
#         I_gray_copy[numRows - i - 1, j] = I_gray[i, j]

# Cut 1/2 image
# for i in range(numRows // 2):
#     for j in range(numCols):
#         I_gray_copy[i, j] = I_gray[i, j]

# Reverse Image by horizontal
for i in range(numRows):
    for j in range(numCols):
        I_gray_copy[i, numCols - j - 1] = I_gray[i, j]

plt.imshow(I_gray_copy, cmap='gray')
plt.show()