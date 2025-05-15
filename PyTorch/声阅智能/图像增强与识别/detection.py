import cv2
import os
import numpy as np
from ultralytics import YOLO


def main(image):
    # 加载本地YOLO模型
    model = YOLO('D:/pycharm/python项目/PyTorch/Runs/Library/best.pt').to('cuda')

    # 进行目标检测推理
    results = model(image, conf=0.5)[0]  # conf设置置信度阈值

    # 创建保存目录
    os.makedirs('detection_results', exist_ok=True)
    os.makedirs('detection_results/cropped_objects', exist_ok=True)

    # 获取带标注的图像（RGB格式）
    annotated_image = results.plot()

    # 转换颜色空间用于OpenCV保存
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # 保存标注后的完整图像
    cv2.imwrite(f'detection_results/annotated.jpg', annotated_image_bgr)

    # 提取OBB检测信息
    obbs = results.obb.data.cpu().numpy()  # 将张量转换为NumPy数组
    print(results.obb.data)

    # 遍历每个检测到的OBB
    cnt = 0
    for i, obb in enumerate(obbs):
        # 解析OBB信息
        # 假设 obb 的结构为 [x, y, w, h, angle, confidence, class_id]
        if len(obb) >= 7:
            x, y, w, h, angle, confidence, class_id = obb[:7]
        else:
            print(f"OBB does not have enough values: {obb}")
            continue

        # 获取类别名称
        class_name = model.names[int(class_id)]
        if class_name != 'book':
            continue

        # 创建旋转矩阵
        center = (int(x), int(y))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 获取旋转后的边界框尺寸
        rotated_w = int(w)
        rotated_h = int(h)

        # 计算旋转后的边界框的四个顶点
        vertices = np.array([
            [x - w/2, y - h/2],
            [x + w/2, y - h/2],
            [x + w/2, y + h/2],
            [x - w/2, y + h/2]
        ])

        # 应用旋转矩阵
        rotated_vertices = cv2.transform(np.array([vertices]), rotation_matrix)[0]

        # 获取旋转后的边界框的最小外接矩形
        x_min = int(np.min(rotated_vertices[:, 0]))
        y_min = int(np.min(rotated_vertices[:, 1]))
        x_max = int(np.max(rotated_vertices[:, 0]))
        y_max = int(np.max(rotated_vertices[:, 1]))

        # 裁剪旋转后的区域
        crop = image[y_min:y_max, x_min:x_max]

        # 跳过空区域
        if crop.size == 0:
            continue

        # 生成文件名：类别_序号_置信度.jpg
        filename = f"{class_name}_{cnt}_conf{confidence:.2f}.jpg"

        # 保存裁剪的实例
        save_path = os.path.join('detection_results/cropped_objects', filename)
        cv2.imwrite(save_path, crop)
        cnt += 1
    print(f"Cropped {cnt} objects")


if __name__ == '__main__':
    image = cv2.imread(r"F:\Deep Learning Datasets\Library_Book\20250325_识别图_jpg\20250325_P6\3架B面4列1层 (1).jpg")
    main(image)