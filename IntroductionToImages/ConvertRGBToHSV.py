import cv2
def f_rgb_to_hsv(r, g, b, scale_factor):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 0) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    if h < 0:
        h += 360
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax )* scale_factor
    v = cmax * scale_factor
    return h, s, v

if __name__ == '__main__':
    print(f_rgb_to_hsv(100, 200, 50, 100))

    img = cv2.imread(r'C:\Users\DELL INSPIRON 5530\PycharmProjects\PythonProject\OpenCV\Img\rgb_image.jpg')
    HsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(type(img))
    print(HsvImg.shape)