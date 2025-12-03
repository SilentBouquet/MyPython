import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# 读取图像
image_path = 'image.jpg'
image = cv2.imread(image_path, 0)  # 以灰度模式读取图像

# 图像预处理：将图像转换为浮点数并归一化
image_float = image.astype(np.float32) / 255.0


# 傅里叶变换
def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)  # 防止log(0)
    return fshift, magnitude_spectrum


# 逆傅里叶变换
def compute_ifft(fshift):
    ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(ishift)
    img_back = np.abs(img_back)
    return img_back


# 创建理想低通滤波器（ILL）
def create_ill_mask(rows, cols, D0):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d < D0:
                mask[i, j] = 0
    return mask


# 创建高斯低通滤波器（GLL）
def create_gll_mask(rows, cols, D0):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = np.exp(- (d**2) / (2 * (D0**2)))
    return mask


# 创建巴特沃斯低通滤波器（BLL）
def create_bll_mask(rows, cols, D0, n):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1 / (1 + (d / D0)**(2 * n))
    return mask


# 主函数
def main():
    rows, cols = image.shape
    D0_values = [20, 40, 60]  # 不同的截止频率
    n_butterworth = 2  # 巴特沃斯滤波器的阶数

    # 计算原始图像的傅里叶变换
    fshift, magnitude_spectrum = compute_fft(image_float)

    # 绘制原始图像和频谱
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('频谱幅度')
    plt.axis('off')
    plt.show()

    # 为每个截止频率创建一个独立的窗口
    for D0 in D0_values:
        # 创建理想低通滤波器
        ill_mask = create_ill_mask(rows, cols, D0)
        fshift_ill = fshift * ill_mask
        img_back_ill = compute_ifft(fshift_ill)

        # 创建高斯低通滤波器
        gll_mask = create_gll_mask(rows, cols, D0)
        fshift_gll = fshift * gll_mask
        img_back_gll = compute_ifft(fshift_gll)

        # 创建巴特沃斯低通滤波器
        bll_mask = create_bll_mask(rows, cols, D0, n_butterworth)
        fshift_bll = fshift * bll_mask
        img_back_bll = compute_ifft(fshift_bll)

        # 绘制滤波器掩模
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(ill_mask, cmap='gray')
        plt.title(f'ILL Mask D0={D0}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gll_mask, cmap='gray')
        plt.title(f'GLL Mask D0={D0}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(bll_mask, cmap='gray')
        plt.title(f'BLL Mask D0={D0}\nn={n_butterworth}')
        plt.axis('off')
        plt.show()

        # 绘制滤波后的图像
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img_back_ill, cmap='gray')
        plt.title(f'ILL D0={D0}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_back_gll, cmap='gray')
        plt.title(f'GLL D0={D0}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img_back_bll, cmap='gray')
        plt.title(f'BLL D0={D0}\nn={n_butterworth}')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    main()