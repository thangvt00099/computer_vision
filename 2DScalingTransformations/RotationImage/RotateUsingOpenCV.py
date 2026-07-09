import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    grayImage = r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein.webp"
    I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)

    # Lấy tâm của ảnh
    image_center = tuple(np.array(I_gray.shape[1::-1]) / 2)
    a = 90
    scale = 1.0

    # ma trận xoay
    rot_mat = cv2.getRotationMatrix2D(image_center, a, scale)

    # Sử dụng phép biến đổi để ra ảnh đích
    I2 = cv2.warpAffine(I_gray, rot_mat, I_gray.shape[::-1])
    plt.imshow(I2, cmap='gray')
    plt.show()