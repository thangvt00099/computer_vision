import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    im1 = cv2.imread(r"C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\1.jpg")
    im2 = cv2.imread(r"C:\Users\DELL INSPIRON "
                     r"5530\PycharmProjects\PythonProject\OpenCV\Img\2.jpg")
    im3 = cv2.imread(r"C:\Users\DELL INSPIRON "
                     r"5530\PycharmProjects\PythonProject\OpenCV\Img\3.jpg")

    # B1: Tạo danh sách ảnh
    imgs = []
    imgs.append(im1)
    imgs.append(im2)
    imgs.append(im3)

    # B2: Tạo object Stitcher để chuẩn bị Panoramic ảnh
    # cv2.Sticher_PANORAMA: tự động sắp xếp ảnh theo thứ tự
    # cv2.Sticher_SCANS: dùng khi ảnh đã được sắp xếp sẵn thứ tự
    M = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    # B3: Sử dụng stitch() - thực hiện ghép ảnh
    # status - mã trạng thái (thành công hay lỗi)
    # result - ảnh toàn cảnh kết quả
    status, result = M.stitch(imgs)

    # plt.subplot(131)
    # plt.imshow(im1[:,:,::-1])
    # plt.subplot(132)
    # plt.imshow(im2[:,:,::-1])
    # plt.subplot(133)
    # plt.imshow(im3[:,:,::-1])

    plt.imshow(result[:,:,::-1])
    plt.show()
