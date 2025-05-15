import os
import json
import argparse
from collections import defaultdict


def convert_dataset(images_dir, gts_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化标注信息
    detection_labels = defaultdict(list)
    recognition_labels = defaultdict(list)

    # 遍历所有图片和对应的标注文件
    for image_file in os.listdir(images_dir):
        image_name = os.path.splitext(image_file)[0]
        gt_file = os.path.join(gts_dir, f"{image_name}.txt")

        # 跳过没有对应标注文件的图片
        if not os.path.exists(gt_file):
            print(f"Warning: No ground truth file found for image {image_file}")
            continue

        # 读取标注文件
        with open(gt_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        for line in lines:
            # 解析标注信息
            parts = line.strip().split(',')
            if len(parts) < 10:
                print(f"Warning: Invalid format in line: {line}")
                continue

            # 提取坐标和文本
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            difficult = parts[8]
            transcript = ','.join(parts[9:]).strip('"')

            # 添加到检测标注中
            detection_labels[image_file].append({
                "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "transcription": transcript,
                "difficult": difficult
            })

            # 添加到识别标注中
            recognition_labels[image_file].append(transcript)

    # 保存检测标注（修改部分：每行一个图片的标注信息）
    with open(os.path.join(output_dir, 'train_label.txt'), 'w', encoding='utf-8') as f:
        for image_file, annotations in detection_labels.items():
            f.write(f"{image_file}\t{json.dumps(annotations, ensure_ascii=False)}\n")

    # 保存识别标注（无需修改，格式已经正确）
    with open(os.path.join(output_dir, 'recognition_label.txt'), 'w', encoding='utf-8') as f:
        for image_file, transcripts in recognition_labels.items():
            for transcript in transcripts:
                f.write(f"{image_file}\t{transcript}\n")

    print(f"Conversion completed. Detection labels saved to {os.path.join(output_dir, 'train_label.txt')}")
    print(f"Recognition labels saved to {os.path.join(output_dir, 'recognition_label.txt')}")


if __name__ == "__main__":
    # 修改以下路径为你自己的路径
    images_dir = r"F:\Deep Learning Datasets\RCTW-17\train_images"
    gts_dir = r'F:\Deep Learning Datasets\RCTW-17\train_gts'
    output_dir = 'Paddle_data'

    convert_dataset(images_dir, gts_dir, output_dir)