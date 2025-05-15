from PIL import Image, ImageEnhance, ImageFilter
import os


def preprocess_image_pillow(image_path):
    """
    使用Pillow进行图像预处理
    """
    # 打开图片
    image = Image.open(image_path)

    # 转为灰度图
    image = image.convert('L')

    # 增强对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # 锐化图片
    image = image.filter(ImageFilter.SHARPEN)

    # 去噪
    image = image.filter(ImageFilter.MedianFilter())

    return image


def process_images_in_folder(folder_path, output_dir, use_pillow=True):
    """
    处理文件夹中的所有图片
    """
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)

            # 使用Pillow进行预处理
            processed_image = preprocess_image_pillow(image_path)

            # 构建输出路径
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_image_path = os.path.join(output_dir, f"{base_name}_processed.jpg")

            # 保存预处理后的图片
            processed_image.save(processed_image_path)
            print(f"预处理后的图片已保存到: {processed_image_path}")


if __name__ == "__main__":
    folder_path = r"F:\Deep Learning Datasets\Library_Book\images"
    output_dir = r"F:\Deep Learning Datasets\Library_Book\images_processed"

    # 处理文件夹中的所有图片
    process_images_in_folder(folder_path, output_dir)