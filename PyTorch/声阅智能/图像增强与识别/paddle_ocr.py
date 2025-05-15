import cv2
from paddleocr import PaddleOCR, draw_ocr


def ocr_image(image_path, output_path=None):
    """
    使用PaddleOCR进行文字检测和识别
    :param image_path: 输入图片路径
    :param output_path: 可选，保存结果图片的路径
    :return: 识别到的文字及其位置
    """
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)  # 使用中文模型，也可以设置为"en"使用英文模型

    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法加载图片 {image_path}")
        return None

    # 进行文字检测和识别
    result = ocr.ocr(img, cls=True)

    # 提取结果
    boxes = [line[0] for line in result[0]]  # 文字框的坐标
    texts = [line[1][0] for line in result[0]]  # 识别到的文字
    scores = [line[1][1] for line in result[0]]  # 识别的置信度

    # 打印结果
    print("识别结果：")
    for box, text, score in zip(boxes, texts, scores):
        print(f"文字: {text}, 置信度: {score:.2f}, 位置: {box}")

    # 可视化结果（在图片上绘制文字框和文字）
    if output_path:
        img = draw_ocr(img, boxes=boxes, texts=texts, scores=scores)
        cv2.imwrite(output_path, img)
        print(f"结果已保存到 {output_path}")

    return texts, boxes, scores


if __name__ == "__main__":
    # 输入图片路径
    image_path = "example.jpg"  # 替换为你的图片路径
    # 输出结果图片路径（可选）
    output_path = "result.jpg"  # 替换为你想要保存的路径

    # 执行OCR
    texts, boxes, scores = ocr_image(image_path, output_path)