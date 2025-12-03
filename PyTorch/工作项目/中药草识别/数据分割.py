import os
import shutil
import random
import json

# 数据集主目录
dataset_dir = 'data'  # 数据集目录

# 输出目录
output_train_dir = 'train'
output_val_dir = 'val'
output_test_dir = 'test'

# 分割比例（训练集 : 验证集 : 测试集）
split_ratio = (0.7, 0.2, 0.1)

# 中文类别映射字典
class_mapping = {
    "baihe": "百合",
    "dangshen": "党参",
    "gouqi": "枸杞",
    "huaihua": "槐花",
    "jinyinhua": "金银花"
}

# 创建输出目录
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 遍历每个药草类别文件夹
for idx, class_name in enumerate(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 获取该类别的所有图片
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # 随机打乱图片顺序

    # 计算每个数据集的图片数量
    num_images = len(images)
    num_train = int(num_images * split_ratio[0])
    num_val = int(num_images * split_ratio[1])
    num_test = num_images - num_train - num_val

    # 分割图片
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # 创建每个类别的输出子目录
    train_class_dir = os.path.join(output_train_dir, class_name)
    val_class_dir = os.path.join(output_val_dir, class_name)
    test_class_dir = os.path.join(output_test_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # 复制图片到相应的输出目录
    for img in train_images:
        shutil.copy(img, train_class_dir)
    for img in val_images:
        shutil.copy(img, val_class_dir)
    for img in test_images:
        shutil.copy(img, test_class_dir)

# 保存分类映射字典到 JSON 文件
with open('class_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, ensure_ascii=False)

print("数据集分割完成！")
print(f"训练集样本数：{sum(len(os.listdir(os.path.join(output_train_dir, class_name))) for class_name in os.listdir(dataset_dir))}")
print(f"验证集样本数：{sum(len(os.listdir(os.path.join(output_val_dir, class_name))) for class_name in os.listdir(dataset_dir))}")
print(f"测试集样本数：{sum(len(os.listdir(os.path.join(output_test_dir, class_name))) for class_name in os.listdir(dataset_dir))}")