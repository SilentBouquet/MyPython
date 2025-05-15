import cv2
import numpy as np
from PIL import Image, ImageFilter


def enhance_and_denoise_image(input_path, output_path):
    # 打开图片
    image = Image.open(input_path)

    # 放大图片
    new_size = (image.width * 2, image.height * 2)
    image_resized = image.resize(new_size, Image.LANCZOS)  # 使用LANCZOS滤镜放大图片

    # 锐化图片
    sharpened_image = image_resized.filter(ImageFilter.SHARPEN)

    # 转换为NumPy数组以便使用OpenCV
    img_array = np.array(sharpened_image)

    # 如果是灰度图，直接进行去噪
    if len(img_array.shape) == 2:
        denoised_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
    # 如果是彩色图，使用彩色去噪
    else:
        denoised_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # 将NumPy数组转换回PIL图像
    denoised_image = Image.fromarray(denoised_array)

    # 保存结果
    denoised_image.save(output_path)


if __name__ == '__main__':
    input_path = r'detection_results\cropped_objects\book_6_conf0.95.jpg'
    output_path = 'outputs/b_output.jpg'
    enhance_and_denoise_image(input_path, output_path)