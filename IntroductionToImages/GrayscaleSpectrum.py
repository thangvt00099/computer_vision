import numpy as np
import matplotlib.pyplot as plt

# np.arange([start, ], stop, [step, ],...) - trả về các giá trị cách đều nhau trong 1 khoảng
img = np.arange(4)
print(img)
print(img.ndim)

# np.newaxis - thêm 1 chiều cho array
img = img[np.newaxis, :]
print(img.shape)

# np.repeat(arr, repeats, axis) - Lặp lại phần tử của array sau chính nó
img = np.repeat(img, 10, axis=0)
print(img.shape)
print(img)

# plt.imshow # plt.show => imshow hiển thị dữ liệu như 1 ảnh
plt.imshow(img, cmap='gray')
plt.show()