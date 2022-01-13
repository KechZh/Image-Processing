import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW3/HW3_test_image/kitchen.jpg'
write_path = 'Image Processing/HW3/HW3_test_image/rgb_en.bmp'
ratio = 2.0

def rgb_enhancement(img, ratio):
    img = np.array(img)
    new_img = np.zeros(img.shape)

    height, width, ch = img.shape

    for i in range(height):
        for j in range(width):
            for k in range(ch):
                new_img[i][j][k] = min(img[i][j][k] * ratio, 255)

    new_img = new_img.astype(np.uint8)

    return new_img

def show_figure(img, new_img):
    plt.figure()

    plt.subplot(1, 2, 1)                       
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 2, 2)                       
    plt.title('rgb_en')
    plt.axis('off')
    plt.imshow(new_img)

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path)

    new_img = rgb_enhancement(img, ratio)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()