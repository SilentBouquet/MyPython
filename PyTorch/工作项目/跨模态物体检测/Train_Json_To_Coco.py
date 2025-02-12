import json
import os
import cv2

input_json_path = r"F:\Deep Learning Datasets\SODA10M\labeled_trainval\SSLAD-2D\labeled\annotations\instance_train.json"
output_coco_path = (r"F:\Deep Learning Datasets\SODA10M\labeled_trainval\SSLAD-2D\labeled\annotations_coco"
                    r"\instance_train_coco.json")


# 从图像文件中获取尺寸信息
def get_image_info(image_id, image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": os.path.basename(image_path),
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": "2025-03-01"
    }


def convert_to_coco(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    # 提取 categories 部分
    categories = data["categories"]
    annotations = data["annotations"]

    # 创建 COCO 格式的基础结构
    coco_data = {
        "info": {
            "description": "Soda10M Dataset in COCO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "Your Name",
            "date_created": "2023-10-10"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_ids = [i for i in range(1, 5001)]

    # 为每张图片生成 images 部分
    for image_id in image_ids:
        name_id = str(image_id)
        formatted_number = name_id.zfill(6)
        image_path = (f"F:/Deep Learning Datasets/SODA10M/labeled_trainval/"
                      f"SSLAD-2D/labeled/train/HT_TRAIN_" + formatted_number + "_SH_000.jpg")  # 替换为你的实际图像路径
        print(image_path)
        image_info = get_image_info(image_id, image_path)
        coco_data["images"].append(image_info)
        print("已完成{}个图片信息保存".format(image_id))

    # 为每个标注生成 annotations 部分
    cnt = 1
    for anno in annotations:
        coco_anno = {
            "id": anno["id"],
            "image_id": anno["image_id"],
            "category_id": anno["category_id"],
            "bbox": anno["bbox"],
            "segmentation": [],
            "area": anno["area"],
            "iscrowd": anno["iscrowd"],
            "ignore": 0
        }
        coco_data["annotations"].append(coco_anno)
        print("已完成{}/{}个标注信息转换".format(cnt, len(annotations)))
        cnt += 1

    # 保存生成的 COCO 格式数据
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)


if __name__ == "__main__":
    convert_to_coco(input_json_path, output_coco_path)
    print("Done!")