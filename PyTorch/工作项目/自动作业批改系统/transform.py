import os

# 定义输入文件夹和输出文件夹
input_folder = r"D:\LabelImg\tiankong"  # 输入文件夹路径
output_folder = r"D:\LabelImg\tiankong_img"  # 输出文件夹路径

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有文件
files = os.listdir(input_folder)

# 筛选出图片文件（假设图片扩展名为 .jpg、.png、.jpeg 等）
image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 遍历图片文件并重命名
for index, filename in enumerate(image_files, start=1):
    # 指定输入文件和输出文件路径
    input_path = os.path.join(input_folder, filename)
    output_filename = f"train{index}.png"  # 修改为所需的图片格式
    output_path = os.path.join(output_folder, output_filename)

    # 复制文件并重命名
    os.rename(input_path, output_path)
    print(f"Renamed: {filename} -> {output_filename}")

print("All done!")