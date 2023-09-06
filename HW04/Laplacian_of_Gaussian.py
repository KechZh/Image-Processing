import math
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW4/HW4_test_image/image1.jpg'
write_path = 'Image Processing/HW4/HW4_test_image/LOG.jpg'
size = 9
sigma = 1.4
time = 50

def make_log_filter(sigma, size, time):
    mask = np.zeros([size, size])

    mid = size // 2

    for i in range(-mid, mid + 1):
        for j in range(-mid, mid + 1):
            tmp = (i ** 2 + j ** 2) / (2 * sigma ** 2)

            mask[mid + i][mid + j] = -((1 - tmp) * (math.e ** -tmp)) / (math.pi * sigma ** 4) * time

    return mask

def image_filter(img, mask):
    img = np.array(img)
    new_img = np.zeros(img.shape)

    height, width = img.shape
    mid = mask.shape[0] // 2

    tmp1 = np.zeros((height + mid * 2, width + mid * 2))
    tmp2 = np.zeros((height + mid * 2, width + mid * 2))

    for i in range(height):
        for j in range(width):
            tmp1[i + mid][j + mid] = img[i][j]

    for i in range(mid, height + mid):
        for j in range(mid, width + mid):
            for x in range(-mid, mid + 1):
                for y in range(-mid, mid + 1):
                    tmp2[i][j] += tmp1[i + x][j + y] * mask[mid + x][mid + y]

    for i in range(height):
        for j in range(width):
            new_img[i][j] = tmp2[i + mid][j + mid]

    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    plt.subplot(1, 2, 2)                       
    plt.title('LOG')
    plt.axis('off')
    plt.imshow(new_img, cmap = 'gray')

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)

    mask = make_log_filter(sigma, size, time)

    new_img = image_filter(img, mask)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()