import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW1/HW1_test_image/Peppers.bmp'
write_path = 'Image Processing/HW1/HW1_test_image/GT.bmp'
r = 2

def gamma_transformation(img, r):
    img = np.array(img)
    new_img = np.zeros(img.shape)

    height, width = img.shape

    for i in range(height):
        for j in range(width):
            new_img[i][j] = (img[i][j] / 255) ** r * 255

    new_img = new_img.astype(np.uint8)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    plt.subplot(1, 2, 2)                       
    plt.title('GT')
    plt.axis('off')
    plt.imshow(new_img, cmap = 'gray')

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)

    new_img = gamma_transformation(img, r)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()