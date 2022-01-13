import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

read_path = 'Image Processing/HW4/HW4_test_image/image1.jpg'
write_path = 'Image Processing/HW4/HW4_test_image/SO.jpg'
mask_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
mask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

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

def sobel_operator(img, mask_x, mask_y):
    edge_x = image_filter(img, mask_x)
    edge_y = image_filter(img, mask_y)

    new_img = np.zeros(img.shape)

    height, width = img.shape

    for i in range(height):
        for j in range(width):
            new_img[i][j] = (edge_x[i][j] ** 2 + edge_y[i][j] ** 2) ** 0.5

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
    plt.title('SO')
    plt.axis('off')
    plt.imshow(new_img, cmap = 'gray')

    plt.show()
    plt.close()

def main():
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)

    new_img = sobel_operator(img, mask_x, mask_y)

    cv2.imwrite(write_path, new_img)

    show_figure(img, new_img)

if __name__ == '__main__':
    main()