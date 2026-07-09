import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Hàm để hiện thị kích thước thật của ảnh khi scale
def displayImageInActualSize(I):
    dpi = mpl.rcParams['figure.dpi'] # dpi (dots per inch - mật độ điểm ảnh khi hiển thị hình)
    H, W = I.shape # grayImage
    # H, W = I[0:2] # colorImage
    fig_size = W / float(dpi), H / float(dpi)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(I, cmap='gray')
    plt.show()

grayImage = r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\modifier_albert_einstein.jpg"

I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)
numRows = I_gray.shape[0]
numCols = I_gray.shape[1]

# Scaling Matrix
S = np.array([[2, 0], [0, 2]])

# Destination Image
I2 = np.zeros((2 * numRows, 2 * numCols), dtype='uint8')

# Forward Transformation - Biến đổi thuận
for i in range(numRows):
    for j in range(numCols):
        P = np.array([i, j])
        P_dash = S.dot(P)
        new_i, new_j =  P_dash[0], P_dash[1]
        I2[new_i, new_j] = I_gray[i, j]

# Tính ma trận khả nghịch của S -> S^-1
Tinv = np.linalg.inv(S)

# Lặp lại destination image
for new_i in range(I2.shape[0]):
    for new_j in range(I2.shape[1]):
        P_dash = np.array([new_i, new_j])
        P = Tinv.dot(P_dash)

        # Tìm nearest neighbor interpolation - tìm điểm lận cận nhất
        # Nếu sử dụng round -> Error:  out of range vì round làm tròn
        P = np.int16(np.floor(P))
        i, j = P[0], P[1]
        I2[new_i, new_j] = I_gray[i, j]

displayImageInActualSize(I2)