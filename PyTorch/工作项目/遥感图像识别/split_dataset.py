import os
import shutil
import random

# 源数据集路径
source_dataset_path = r"F:\Deep Learning Datasets\NWPU VHR-10 dataset\NWPU VHR-10 dataset"
source_positive_image_set_path = os.path.join(source_dataset_path, "positive image set")
source_ground_truth_path = os.path.join(source_dataset_path, "ground truth")
source_yolo_labels_path = os.path.join(source_dataset_path, "yolo_labels")

# 目标数据集路径
target_dataset_path = r"D:\pycharm\python项目\PyTorch\ultralytics-8.3.91\ultralytics\datasets"
target_images_path = os.path.join(target_dataset_path, "images")
target_labels_path = os.path.join(target_dataset_path, "labels")
target_train_images_path = os.path.join(target_images_path, "train")
target_train_labels_path = os.path.join(target_labels_path, "train")
target_val_images_path = os.path.join(target_images_path, "val")
target_val_labels_path = os.path.join(target_labels_path, "val")

# 创建目标数据集目录
os.makedirs(target_train_images_path, exist_ok=True)
os.makedirs(target_train_labels_path, exist_ok=True)
os.makedirs(target_val_images_path, exist_ok=True)
os.makedirs(target_val_labels_path, exist_ok=True)

# 获取所有正样本图像
image_files = os.listdir(source_positive_image_set_path)
random.shuffle(image_files)  # 打乱图像顺序

# 分割比例（训练集:验证集 = 8:2）
train_ratio = 0.8  # 训练集比例
train_size = int(len(image_files) * train_ratio)

# 分割数据集
train_images = image_files[:train_size]
val_images = image_files[train_size:]


# 复制图像和标签到对应的数据集文件夹
def copy_files(image_list, src_images_path, src_labels_path, dst_images_path, dst_labels_path):
    for image_name in image_list:
        image_path = os.path.join(src_images_path, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(src_labels_path, label_name)

        shutil.copy(image_path, dst_images_path)
        shutil.copy(label_path, dst_labels_path)


# 复制训练集
copy_files(train_images, source_positive_image_set_path, source_yolo_labels_path, target_train_images_path, target_train_labels_path)

# 复制验证集
copy_files(val_images, source_positive_image_set_path, source_yolo_labels_path, target_val_images_path, target_val_labels_path)

print("数据集分割完成！")