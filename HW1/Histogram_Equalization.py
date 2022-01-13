import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW1/HW1_test_image/Peppers.bmp'
write_path = 'Image Processing/HW1/HW1_test_image/HE.bmp'

def histogram_equalization(img):
    img = np.array(img)
    new_img = np.zeros(img.shape)

    height, width = img.shape

    f = np.zeros(256)
    cdf = np.zeros(256)
    h = np.zeros(256)
    cdf_min = 256 * 256
    cdf_max = 0

    for i in range(height):
        for j in range(width):
            f[img[i][j]] += 1

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + f[i]

    for i in range(256):
        if cdf[i] < cdf_min and cdf[i] > 0:
            cdf_min = cdf[i]

    for i in range(256):
        if cdf[i] > cdf_max and cdf[i] > 0:
            cdf_max = cdf[i]

    for i in range(256):
        h[i] = round((cdf[i] - cdf_min) / (cdf_max - cdf_min) * 255)

    for i in range(height):
        for j in range(width):
            new_img[i][j] = h[img[i][j]]

    new_img = new_img.astype(np.uint8)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(2, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    plt.subplot(2, 2, 2)                       
    plt.title('HE')
    plt.axis('off')
    plt.imshow(new_img, cmap = 'gray')

    plt.subplot(2, 2, 3)
    plt.title('Original Histrogram')
    plt.hist(img.ravel(), bins = 256, range = (0, 255))

    plt.subplot(2, 2, 4)
    plt.title('HE Histrogram')
    plt.hist(new_img.ravel(), bins = 256, range = (0, 255))

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)

    new_img = histogram_equalization(img)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()