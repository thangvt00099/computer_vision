import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Hàm tính nội suy song tính (Bilinear Interpolation)
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

    # Tính linear interpolation (nội suy tuyến tính) cho đoạn cuối
    if top_row >= 0 and bottom_row < I.shape[0] and left_column >= 0 and right_column < I.shape[1]:
        a = weight_left * I[top_row, left_column] + weight_right * I[top_row, right_column]
        b = weight_left * I[bottom_row, left_column] + weight_right * I[bottom_row, right_column]
        g = weight_top * a + weight_bottom * b
        return np.uint8(g)
    else:
        return 0

def get_matrix_scale(scale):
    size = len(scale)
    matrix = np.zeros((size, size))
    for i, _ in enumerate(matrix):
        matrix[i][i] = scale[i]
    return matrix

def f_scaleImage(scale, I_gray):
    # Tính ma trận khả nghịch của S -> S^-1
    numRows = I_gray.shape[0]
    numCols = I_gray.shape[1]
    S = get_matrix_scale(scale)
    I2 = np.zeros((S.shape[0] * numRows, S.shape[1] * numCols), dtype='uint8')
    Tinv = np.linalg.inv(S)
    # Lặp lại destination image
    for new_i in range(I2.shape[0]):
        for new_j in range(I2.shape[1]):
            P_dash = np.array([new_i, new_j])
            P = Tinv.dot(P_dash)
            # Tìm nearest neighbor interpolation - tìm điểm lận cận nhất
            # Nếu sử dụng round -> Error:  out of range vì round làm tròn
            # P = np.int16(np.floor(P))
            i, j = P[0], P[1]
            if i < 0 or i >= numRows or j < 0 or j >= numCols:
                pass
            else:
                g = f_bilinear_interpolation(i, j, I_gray)
                I2[new_i, new_j] = g
    return I2

if __name__ == '__main__':
    grayImage = (r"C:\Users\DELL INSPIRON "
             r"5530\PycharmProjects\PythonProject\OpenCV\Img\albert_einstein.webp")
    I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)

    # S thường là diagonal matrix -> ma trận đường chéo chính
    # S = np.array([[2, 0], [0, 2]])

    # Scaling Gray Image
    I2 = f_scaleImage([2, 2], I_gray)

    colorImage = r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\rgb_image.jpg"
    I_color = cv2.imread(colorImage, cv2.IMREAD_UNCHANGED)


    # Convert BGR -> RGB
    I_color = I_color[:, :, ::-1]
    scale = [2, 2]
    r = f_scaleImage(scale, I_color[:,:,0])
    g = f_scaleImage(scale, I_color[:,:,1])
    b = f_scaleImage(scale, I_color[:,:,2])

    I_result = np.zeros((r.shape[0], r.shape[1], 3), dtype='uint8')
    I_result[:,:,0] = r
    I_result[:,:,1] = g
    I_result[:,:,2] = b

    plt.imshow(I_result)
    plt.show()