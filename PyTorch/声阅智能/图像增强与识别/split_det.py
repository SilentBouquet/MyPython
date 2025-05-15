import os
import json
import random
import shutil


def split_dataset(images_dir, detection_labels_path, recognition_labels_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    det_dir = os.path.join(output_dir, 'det')
    rec_dir = os.path.join(output_dir, 'rec')
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    # 获取所有图片文件名
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(image_files)

    # 计算分割点
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 分割数据集
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]

    # 创建数据集目录
    datasets = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    # 处理检测数据
    for dataset_name, image_list in datasets.items():
        dataset_dir = os.path.join(det_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 复制图片
        for image in image_list:
            src_path = os.path.join(images_dir, image)
            dst_path = os.path.join(dataset_dir, image)
            shutil.copy(src_path, dst_path)

    # 分割检测标注文件
    with open(detection_labels_path, 'r', encoding='utf-8') as f:
        detection_labels = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            image_file, annotations = parts
            detection_labels[image_file] = json.loads(annotations)

    for dataset_name, image_list in datasets.items():
        with open(os.path.join(det_dir, f'{dataset_name}_label.txt'), 'w', encoding='utf-8') as f:
            for image_file in image_list:
                if image_file in detection_labels:
                    annotations = detection_labels[image_file]
                    f.write(f"{image_file}\t{json.dumps(annotations, ensure_ascii=False)}\n")

    # 处理识别数据
    for dataset_name, image_list in datasets.items():
        dataset_dir = os.path.join(rec_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 复制图片
        for image in image_list:
            src_path = os.path.join(images_dir, image)
            dst_path = os.path.join(dataset_dir, image)
            shutil.copy(src_path, dst_path)

    # 分割识别标注文件
    with open(recognition_labels_path, 'r', encoding='utf-8') as f:
        recognition_labels = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            image_file, transcript = parts
            if image_file not in recognition_labels:
                recognition_labels[image_file] = []
            recognition_labels[image_file].append(transcript)

    for dataset_name, image_list in datasets.items():
        with open(os.path.join(rec_dir, f'recognition_{dataset_name}_label.txt'), 'w', encoding='utf-8') as f:
            for image_file in image_list:
                if image_file in recognition_labels:
                    transcripts = recognition_labels[image_file]
                    for transcript in transcripts:
                        f.write(f"{image_file}\t{transcript}\n")

    print(f"Dataset split completed.")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")


if __name__ == "__main__":
    # 修改以下路径为你自己的路径
    images_dir = r"F:\Deep Learning Datasets\RCTW-17\train_images"
    detection_labels_path = r'Paddle_data\train_label.txt'
    recognition_labels_path = r'Paddle_data\recognition_label.txt'
    output_dir = r'D:\pycharm\python项目\PyTorch\PaddleOCR-main\train_data'

    split_dataset(images_dir, detection_labels_path, recognition_labels_path, output_dir)