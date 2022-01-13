import math
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW3/HW3_test_image/kitchen.jpg'
write_path = 'Image Processing/HW3/HW3_test_image/hsi_en.bmp'
ratio = 2.0

def rgb_to_hsi(rgb_img):
    rgb_img = np.array(rgb_img)
    hsi_img = np.zeros(rgb_img.shape)

    height, width, tmp = rgb_img.shape

    for x in range(height):
        for y in range(width):
            r = rgb_img[x][y][0] / 255
            g = rgb_img[x][y][1] / 255
            b = rgb_img[x][y][2] / 255

            if r == g and g == b:
                h = 0
                s = 0
            else:
                if b <= g:
                    h = math.acos(((r - g) + (r - b)) * 0.5 / ((r - g) ** 2 + (r - b) * (g - b)) ** 0.5)
                else:
                    h = math.pi * 2 - math.acos(((r - g) + (r - b)) * 0.5 / ((r - g) ** 2 + (r - b) * (g - b)) ** 0.5)

                s = 1 - min([r, g, b]) * 3 / (r + g + b)
            
            i = (r + g + b) / 3

            hsi_img[x][y][0] = h
            hsi_img[x][y][1] = s
            hsi_img[x][y][2] = i

    return hsi_img

def hsi_to_rgb(hsi_img):
    hsi_img = np.array(hsi_img)
    rgb_img = np.zeros(hsi_img.shape)

    height, width, tmp = rgb_img.shape

    for x in range(height):
        for y in range(width):
            h = hsi_img[x][y][0]
            s = hsi_img[x][y][1]
            i = hsi_img[x][y][2]

            if h < math.pi * 2 / 3:
                b = i * (1 - s)
                r = i * (1 + s * math.cos(h) / math.cos(math.pi / 3 - h))
                g = i * 3 - r - b
            elif h < math.pi * 4 / 3:
                h = h - math.pi * 2 / 3
                r = i * (1 - s)
                g = i * (1 + s * math.cos(h) / math.cos(math.pi / 3 - h))
                b = i * 3 - r - g
            else:
                h = h - math.pi * 4 / 3
                g = i * (1 - s)
                b = i * (1 + s * math.cos(h) / math.cos(math.pi / 3 - h))
                r = i * 3 - g - b

            rgb_img[x][y][0] = min(r * 255, 255)
            rgb_img[x][y][1] = min(g * 255, 255)
            rgb_img[x][y][2] = min(b * 255, 255)

    rgb_img = rgb_img.astype(np.uint8)

    return rgb_img

def hsi_enhancement(img, ratio):
    hsi_img = rgb_to_hsi(img)

    height, width, tmp = hsi_img.shape

    for x in range(height):
        for y in range(width):
            hsi_img[x][y][1] = min(hsi_img[x][y][1] * ratio, 1)
            hsi_img[x][y][2] = min(hsi_img[x][y][2] * ratio, 1)

    new_img = hsi_to_rgb(hsi_img)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 2, 2)                       
    plt.title('hsi_en')
    plt.axis('off')
    plt.imshow(new_img)

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path)

    new_img = hsi_enhancement(img, ratio)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()