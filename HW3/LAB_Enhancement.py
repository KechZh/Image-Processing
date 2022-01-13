import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW3/HW3_test_image/kitchen.jpg'
write_path = 'Image Processing/HW3/HW3_test_image/lab_en.bmp'
ratio = 1.5

xn = 0.9515
yn = 1.0000
zn = 1.0886

def fun(t):
    if t > 0.008856:
        ft = t ** (1 / 3)
    else:
        ft = t * 7.787 + (16 / 116)

    return ft

def rgb_to_lab(rgb_img):
    rgb_img = np.array(rgb_img)
    lab_img = np.zeros(rgb_img.shape)

    height, width, tmp = rgb_img.shape

    for i in range(height):
        for j in range(width):
            r = rgb_img[i][j][0] / 255
            g = rgb_img[i][j][1] / 255
            b = rgb_img[i][j][2] / 255

            x = r * 0.412453 + g * 0.357580 + b * 0.180423
            y = r * 0.212671 + g * 0.715160 + b * 0.072169
            z = r * 0.019334 + g * 0.119193 + b * 0.950227

            x = x / xn
            y = y / yn
            z = z / zn

            if y > 0.008856:
                l_star = y ** (1 / 3) * 116 - 16
            else:
                l_star = y * 903.3

            a_star = (fun(x) - fun(y)) * 500
            b_star = (fun(y) - fun(z)) * 200

            lab_img[i][j][0] = l_star
            lab_img[i][j][1] = a_star
            lab_img[i][j][2] = b_star

    return lab_img

def lab_to_rgb(lab_img):
    lab_img = np.array(lab_img)
    rgb_img = np.zeros(lab_img.shape)

    height, width, tmp = lab_img.shape

    for i in range(height):
        for j in range(width):
            l_star = lab_img[i][j][0]
            a_star = lab_img[i][j][1]
            b_star = lab_img[i][j][2]

            y = (l_star + 16) / 116
            x = y + a_star / 500
            z = y - b_star / 200

            if y > 0.008856:
                y = y ** 3 * yn
            else:
                y = (y - 16) / 116 * 3 * 0.008856 ** 2 * yn

            if x > 0.008856:
                x = x ** 3 * xn
            else:
                x = (x - 16) / 116 * 3 * 0.008856 ** 2 * xn

            if z > 0.008856:
                z = z ** 3 * zn
            else:
                z = (z - 16) / 116 * 3 * 0.008856 ** 2 * zn

            r = x * 3.240479 + y * -1.537150 + z * -0.498535
            g = x * -0.969256 + y * 1.875992 + z * 0.041556
            b = x * 0.055648 + y * -0.204043 + z * 1.057311

            rgb_img[i][j][0] = min(r * 255, 255)
            rgb_img[i][j][1] = min(g * 255, 255)
            rgb_img[i][j][2] = min(b * 255, 255)

    rgb_img = rgb_img.astype(np.uint8)

    return rgb_img

def lab_enhancement(img, ratio):
    lab_img = rgb_to_lab(img)

    height, width, tmp = lab_img.shape

    for i in range(height):
        for j in range(width):
            lab_img[i][j][0] = min(lab_img[i][j][0] * ratio, 100)

    new_img = lab_to_rgb(lab_img)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 2, 2)                       
    plt.title('lab_en')
    plt.axis('off')
    plt.imshow(new_img)

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path)

    new_img = lab_enhancement(img, ratio)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()