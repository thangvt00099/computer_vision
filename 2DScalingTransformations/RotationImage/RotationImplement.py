import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Hàm tính nội suy song tính (bilinear interpolation)
def f_bilinear_interpolation(row, col, I):
    # Tọa độ x (cột) của điểm trái và phải
    left_column = int(col)
    right_column = left_column + 1

    # Tính trọng số bên trái và phải
    weight_right = col - left_column
    weight_left = right_column - col
    top_row = int(row)
    bottom_row = top_row + 1
    weight_top = bottom_row - row
    weight_bottom = row - top_row

    # Tính linear interpolation cho đoạn cuối
    if top_row >= 0 and bottom_row < I.shape[0] and left_column >= 0 and right_column < I.shape[1]:
        a = weight_left * I[top_row, left_column] + weight_right * I[top_row, right_column]
        b = weight_left * I[bottom_row, left_column] + weight_right * I[bottom_row, right_column]
        g = weight_top * a + weight_bottom * b
        return np.uint8(g)
    else:
        return 0

if __name__ == '__main__':
    grayImage = r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein.webp"
    I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)

    numRows = I_gray.shape[0]
    numCols = I_gray.shape[1]

    a = 90
    # Tính cos a và đổi sang radian
    cos_a = np.cos(np.deg2rad(a))

    # Tính sin a và đổi sang radian
    sin_a = np.sin(np.deg2rad(a))

    # ma trận xoay
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # rowMax, colMax của source image
    rowMax, colMax = I_gray.shape[0], I_gray.shape[1]

    # Tạo tọa độ 4 điểm
    cords = np.array([[0, 0], [0, colMax - 1], [rowMax - 1, 0], [rowMax - 1, colMax - 1]])
    A_dash = R.dot(cords.T)
    mins = A_dash.min(axis=1)
    maxs = A_dash.max(axis=1)
    minR = np.int64(np.floor(mins[0]))
    minC = np.int64(np.floor(mins[1]))
    maxR = np.int64(np.ceil(maxs[0]))
    maxC = np.int64(np.ceil(maxs[1]))

    # Tính H, W của source image
    H, W = maxR - minR + 1, maxC - minC + 1

    # Tạo destination image zero
    I2 = np.zeros((H, W), dtype='uint8')
    Tinv = np.linalg.inv(R)
    for new_i in range(minR, maxR):
        for new_j in range(minC, maxC):
            P_dash = np.array([new_i, new_j])
            P = Tinv.dot(P_dash)
            i, j = P[0], P[1]
            if i < 0 or i >= numRows or j < 0 or j >= numCols:
                pass
            else:
                g = f_bilinear_interpolation(i, j, I_gray)
                I2[new_i - minR, new_j - minC] = g

    plt.imshow(I2, cmap='gray')
    plt.show()