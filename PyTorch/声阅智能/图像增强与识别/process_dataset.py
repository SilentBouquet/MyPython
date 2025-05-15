import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split


def extract_annotations(image_dir, gt_dir):
    """
    提取标注信息，返回图像路径和对应的文本标注
    """
    cnt1 = 1
    cnt2 = 1
    annotations = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        gt_file = os.path.join(gt_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        if not os.path.exists(gt_file):
            continue

        with open(gt_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue

            # 提取四边形坐标和文本
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
            difficult = parts[8]
            transcript = ','.join(parts[9:]).strip('"')

            # 过滤困难样本
            if difficult == '1':
                continue

            # 提取文本区域
            pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

            # 转换为 PIL 图像
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(f"Warning: Failed to open image {image_path}. Error: {e}")
                continue

            # 裁剪文本区域
            left = min(x1, x2, x3, x4)
            upper = min(y1, y2, y3, y4)
            right = max(x1, x2, x3, x4)
            lower = max(y1, y2, y3, y4)

            if left >= right or upper >= lower:
                continue

            try:
                text_img = img.crop((left, upper, right, lower))
            except Exception as e:
                print(f"Warning: Failed to crop image {image_path}. Error: {e}")
                continue

            # 保存裁剪后的文本图像
            text_img_path = os.path.join('text_images', f"{os.path.splitext(image_file)[0]}_{len(annotations)}.jpg")
            os.makedirs(os.path.dirname(text_img_path), exist_ok=True)

            try:
                text_img.save(text_img_path)
            except Exception as e:
                print(f"Warning: Failed to save image {text_img_path}. Error: {e}")
                continue

            annotations.append((text_img_path, transcript))
            print(f'已保存到{text_img_path}  第{cnt1}张图片 {cnt2}个文字段')
            cnt2 += 1
        cnt1 += 1

    return annotations


# 图像增强
def augment_image(image_path):
    """
    对图像进行增强
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Warning: Failed to open image {image_path}. Error: {e}")
        return img

    # 随机旋转
    if random.random() < 0.5:
        angle = random.randint(-10, 10)
        img = img.rotate(angle, expand=True)

    # 随机缩放
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height))

    # 随机颜色抖动
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)

    return img


def process_paddleocr(annotations, output_dir, epochs=10, batch_size=32):
    """
    使用 PaddleOCR 模型进行训练
    """
    # 划分训练集和验证集
    train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

    # 准备训练数据
    train_data_dir = os.path.join(output_dir, 'train_data')
    val_data_dir = os.path.join(output_dir, 'val_data')
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)

    # 准备训练数据
    with open(os.path.join(train_data_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for img_path, label in train_annotations:
            shutil.copy(img_path, train_data_dir)
            img_name = os.path.basename(img_path)
            f.write(f"{img_name}\t{label}\n")

    # 准备验证数据
    with open(os.path.join(val_data_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for img_path, label in val_annotations:
            shutil.copy(img_path, val_data_dir)
            img_name = os.path.basename(img_path)
            f.write(f"{img_name}\t{label}\n")


# 主函数
if __name__ == "__main__":
    # 数据集路径
    images = r"F:\Deep Learning Datasets\RCTW-17\train_images"
    train_gts = r'F:\Deep Learning Datasets\RCTW-17\train_gts'

    # 提取标注信息
    annotations = extract_annotations(images, train_gts)

    # 划分训练集和测试集
    train_annotations, test_annotations = train_test_split(
        annotations,
        test_size=0.2,  # 测试集占 20%
        random_state=42  # 随机种子，确保结果可复现
    )

    print(f"训练集大小: {len(train_annotations)}")
    print(f"测试集大小: {len(test_annotations)}")

    # 数据增强
    augmented_annotations = []
    for img_path, label in train_annotations:
        augmented_img = augment_image(img_path)
        if augmented_img is not None:
            augmented_img_path = img_path.replace('.jpg', '_aug.jpg')
            os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)
            augmented_img.save(augmented_img_path)
            augmented_annotations.append((augmented_img_path, label))

    # 合并原始数据和增强数据
    train_annotations += augmented_annotations

    # 训练模型
    model_output_dir = 'Paddle_data'
    process_paddleocr(train_annotations, model_output_dir)