from paddleocr import PaddleOCR
import cv2


def ocr_image(image_path):
    # 初始化OCR模型
    # 读取图片
    image = cv2.imread(image_path)
    # 进行文字识别
    result = ocr.ocr(image)
    text = ""
    for line in result:
        for word in line:
            # 提取文字内容
            text += word[-1][0] + " "
    return text


if __name__ == "__main__":
    ocr = PaddleOCR()
    image_path2 = r'outputs\b_output.jpg'
    image_path = r'detection_results\cropped_objects\book_0_conf0.97.jpg'
    image_path3 = r'outputs_04-03_15-04_BSRGAN-Text/b_output_BSRGANText.png'
    # recognized_text = ocr_image(image_path)
    recognized_text2 = ocr_image(image_path2)
    # recognized_text3 = ocr_image(image_path3)
    # print("识别结果：", recognized_text)
    print("识别结果：", recognized_text2)
    # print("识别结果：", recognized_text3)