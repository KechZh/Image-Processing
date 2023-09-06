import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW2/HW2_test_image/skeleton_orig.bmp'
write_path = 'Image Processing/HW2/HW2_test_image/UM.bmp'
mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

def image_filter(img, mask):
    img = np.array(img)
    new_img = np.zeros(img.shape)

    height, width = img.shape

    tmp1 = np.zeros((height + 2, width + 2))
    tmp2 = np.zeros((height + 2, width + 2))

    for i in range(height):
        for j in range(width):
            tmp1[i + 1][j + 1] = img[i][j]

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            for x in range(-1, 2):
                for y in range(-1, 2):
                    tmp2[i][j] += tmp1[i + x][j + y] * mask[1 + x][1 + y]

    for i in range(height):
        for j in range(width):
            new_img[i][j] = tmp2[i + 1][j + 1]

    new_img = new_img.astype(np.uint8)

    return new_img

def unsharp_masking(img, mask):
    blur_img = image_filter(img, mask)

    new_img = img + (img - blur_img)

    new_img = new_img.astype(np.uint8)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    plt.subplot(1, 2, 2)                       
    plt.title('UM')
    plt.axis('off')
    plt.imshow(new_img, cmap = 'gray')

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)

    new_img = unsharp_masking(img, mask)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()