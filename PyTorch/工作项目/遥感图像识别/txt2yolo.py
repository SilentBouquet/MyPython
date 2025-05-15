import os
import cv2

# 数据集路径
dataset_path = r"F:\Deep Learning Datasets\NWPU VHR-10 dataset\NWPU VHR-10 dataset"
positive_image_set_path = os.path.join(dataset_path, "positive image set")
ground_truth_path = os.path.join(dataset_path, "ground truth")
yolo_labels_path = os.path.join(dataset_path, "yolo_labels")

# 创建YOLO标签文件夹
os.makedirs(yolo_labels_path, exist_ok=True)

# 遍历正样本图像和对应的标注文件
for image_name in os.listdir(positive_image_set_path):
    image_path = os.path.join(positive_image_set_path, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        continue
    image_height, image_width, _ = image.shape

    # 获取对应的标注文件名
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(ground_truth_path, label_name)

    # 检查标注文件是否存在
    if not os.path.exists(label_path):
        print(f"标注文件不存在：{label_path}")
        continue

    # 读取标注文件并转换为YOLO格式
    yolo_label_path = os.path.join(yolo_labels_path, label_name)
    with open(label_path, 'r') as f_in, open(yolo_label_path, 'w') as f_out:
        for line in f_in:
            # 去掉括号并分割坐标和类别
            line = line.replace('(', '').replace(')', '')
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            class_id = int(parts[4]) - 1  # 将类别ID转换为从0开始

            # 计算YOLO格式的中心坐标、宽度和高度（归一化）
            center_x = (x1 + x2) / (2 * image_width)
            center_y = (y1 + y2) / (2 * image_height)
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            # 写入YOLO格式的标注
            f_out.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

print("标注转换完成！")