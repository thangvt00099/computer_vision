import cv2
import matplotlib.pyplot as plt
import numpy as np

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

# Hàm lấy rowMax, rowMin, colMax, colMin, H, W của source image
# Matrix 3x3 với hàng cuối là [0, 0, 1]
def f_getExtentsProjective(T, rowMax, colMax):
    cords = np.array([[0, 0, 1], [0, colMax - 1, 1], [rowMax - 1, 0, 1], [rowMax - 1, colMax - 1, 1]])
    A_dash = T.dot(cords.T)
    A_dash = A_dash / A_dash[2, :]
    mins = A_dash.min(axis=1)
    maxs = A_dash.max(axis=1)
    min_row = np.int64(np.floor(mins[0]))
    min_col = np.int64(np.floor(mins[1]))
    max_row = np.int64(np.ceil(maxs[0]))
    max_col = np.int64(np.ceil(maxs[1]))

    # Tính H, W của source image
    H, W = max_row - min_row + 1, max_col - min_col + 1

    return min_row, min_col, max_row, max_col, H, W

# Hàm áp dụng phép biến đổi lên ảnh (áp dụng cho tất cả loại biến đổi)
def f_transformProjective(T, I_gray):
    numRows, numCols = I_gray.shape[0], I_gray.shape[1]
    min_row, min_col, max_row, max_col, H, W = f_getExtentsProjective(T, I_gray.shape[0], I_gray.shape[1])
    Tinv = np.linalg.inv(T)
    I2 = np.zeros((H, W), dtype='uint8')
    for new_i in range(min_row, max_row):
        for new_j in range(min_col, max_col):
            P_dash = np.array([new_i, new_j, 1])
            P = Tinv.dot(P_dash)
            P = P / P[2]
            i, j = P[0], P[1]
            if i < 0 or i >= numRows or j < 0 or j >= numCols:
                pass
            else:
                g = f_bilinear_interpolation(i, j, I_gray)
                I2[new_i - min_row, new_j - min_col] = g
    return I2

if __name__ == '__main__':
    grayScale = r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein.webp"
    I_gray = cv2.imread(grayScale, cv2.IMREAD_GRAYSCALE)

    a = 30
    cos_a = np.cos(np.deg2rad(a))
    sin_a = np.sin(np.deg2rad(a))
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    T = np.array([[1, 0, -300], [0, 1, -400], [0, 0, 1]])
    Sh = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    S = np.array([[1.3, 0, 0], [0, 0.7, 0], [0, 0, 1]])
    A = np.linalg.inv(T).dot(S).dot(R).dot(Sh).dot(T)
    H = A
    H[2, :] = np.array([0.0001, 0.0001, 1])

    I2 = f_transformProjective(H, I_gray)
    cv2.imwrite(r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein_projective.jpg", I2)
    plt.imshow(I2, cmap='gray')
    plt.show()