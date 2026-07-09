import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img_gray = cv2.imread(r"C:\Users\DELL INSPIRON "
                        r"5530\PycharmProjects\PythonProject\OpenCV\Img"
                          r"\albert_einstein.webp", cv2.IMREAD_GRAYSCALE)
    img_binary = cv2.imread(r"C:\Users\DELL INSPIRON "
                          r"5530\PycharmProjects\PythonProject\OpenCV\Img\j"
                            r".png", cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_binary, cmap='gray')

    # chuyển ảnh gray sang nhị phân bằng cách so sánh với 1 ngưỡng
    th = 150
    im2 = (img_gray > th).astype(np.uint8)
    plt.imshow(im2, cmap='gray')

    # Đảo ngược bit 0/1 thành 1/0
    plt.imshow(~(im2), cmap='gray')
    plt.show()

    # Lấy giá trị duy nhất của mảng - kiểm tra xem ảnh nhị phân có đúng có
    # giá trị 0/1 không
    print(np.unique(img_binary))
    print(np.unique(im2))