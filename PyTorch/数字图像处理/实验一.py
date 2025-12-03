import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像
image = cv2.imread('image.jpg', 0)


# 直方图均衡化
def histogram_equalization(img):
    # 计算原始图像的直方图
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 直方图均衡化处理
    equ_img = cv2.equalizeHist(img)

    # 计算均衡化后图像的直方图
    hist_equ = cv2.calcHist([equ_img], [0], None, [256], [0, 256])

    return equ_img, hist_original, hist_equ


# 加权平均滤波器
def weighted_average_filter(img):
    # 定义加权平均滤波器模板
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
    # 加权平均滤波处理
    mean_img = cv2.filter2D(img, -1, kernel)
    return mean_img


# 中值滤波器
def median_filter(img, kernel_size):
    # 中值滤波处理
    median_img = cv2.medianBlur(img, kernel_size)
    return median_img


# 高斯滤波器（自定义实现）
def gaussian_filter(img, kernel_size, sigma):
    # 生成高斯滤波器模板
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()
    # 高斯滤波处理
    gauss_img = cv2.filter2D(img, -1, kernel)
    return gauss_img


# 拉普拉斯滤波器（自定义实现）
def laplacian_filter(img):
    # 定义拉普拉斯滤波器模板
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    # 拉普拉斯滤波处理
    laplacian_img = cv2.filter2D(img, -1, kernel)
    return laplacian_img


# 主函数
def main():
    global image

    # 直方图均衡化
    equ_img, hist_original, hist_equ = histogram_equalization(image)

    # 空域滤波
    mean_img = weighted_average_filter(image)
    median_img = median_filter(image, 3)
    gauss_img = gaussian_filter(image, 5, 1)
    laplacian_img = laplacian_filter(image)

    # 绘制原始图像及其直方图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.plot(hist_original)
    plt.title('Original Histogram')
    plt.tight_layout()
    plt.show()

    # 绘制均衡化后的图像及其直方图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(equ_img, cmap='gray')
    plt.title('Equalized Image')
    plt.subplot(1, 2, 2)
    plt.plot(hist_equ)
    plt.title('Equalized Histogram')
    plt.tight_layout()
    plt.show()

    # 绘制加权平均滤波后的图像
    plt.figure()
    plt.imshow(mean_img, cmap='gray')
    plt.title('Weighted Average Filtered Image')
    plt.show()

    # 绘制中值滤波后的图像
    plt.figure()
    plt.imshow(median_img, cmap='gray')
    plt.title('Median Filtered Image')
    plt.show()

    # 绘制高斯滤波后的图像
    plt.figure()
    plt.imshow(gauss_img, cmap='gray')
    plt.title('Gaussian Filtered Image')
    plt.show()

    # 绘制拉普拉斯滤波后的图像
    plt.figure()
    plt.imshow(laplacian_img, cmap='gray')
    plt.title('Laplacian Filtered Image')
    plt.show()


if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    main()