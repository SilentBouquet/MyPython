import os
import xml.etree.ElementTree as ET

# 定义类别名称和类别索引
categories = {
    "answer": 0,
    "question": 1
}

# 输入和输出路径
input_folder = r'D:\LabelImg\Saved'
output_folder = r"D:\LabelImg\Saved_Yolo"

os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有 XML 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".xml"):
        print(filename)
        xml_path = os.path.join(input_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片宽高
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        yolo_lines = []

        # 遍历所有的标注对象
        for obj in root.findall("object"):
            # 获取类别名称和边界框坐标
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # 计算 YOLO 格式的中点和宽高
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # 获取类别索引
            class_index = categories.get(name, -1)

            # 如果类别未找到，则跳过
            if class_index == -1:
                print(f"Warning: Category '{name}' not found for file {filename}")
                continue

            # 添加到 YOLO 格式的内容中
            yolo_line = f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_lines.append(yolo_line)

        # 保存为 .txt 文件
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, "w") as f:
            for line in yolo_lines:
                f.write(line + "\n")

print("转换完成！")