import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Hàm để hiện thị kích thước thật của ảnh khi scale
def displayImageInActualSize(I):
    dpi = mpl.rcParams['figure.dpi'] # dpi (dots per inch - mật độ điểm ảnh khi hiển thị hình)
    H, W = I.shape
    fig_size = W / float(dpi), H / float(dpi)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(I, cmap='gray')
    plt.show()

grayImage = r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg'
I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)
numRows = I_gray.shape[0]
numCols = I_gray.shape[1]

# Scaling matrix
S = np.array([[2, 0], [0, 2]])

# Destination Image bị tối màu vì các giá trị khi khởi tạo bằng 0
# I2 = np.zeros((2 * numRows, 2 * numCols), dtype='uint8')

# Tăng độ sáng của ảnh bằng cách khởi tạo các giá trị pixel = 255
I2 = np.ones((2 * numRows, 2 * numCols), dtype='uint8') * 255

# Forward Transformation - biến đổi thuận
# Source Image -> Destination Image | P' = SP
for i in range(numRows):
    for j in range(numCols):
        P = np.array([i, j])
        P_dash = S.dot(P)
        new_i, new_j = P_dash[0], P_dash[1]
        I2[new_i, new_j] = I_gray[i, j]

displayImageInActualSize(I2)