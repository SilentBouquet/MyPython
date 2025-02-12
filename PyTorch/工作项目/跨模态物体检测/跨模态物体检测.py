import os
import cv2
import torch
import spacy
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# 定义颜色关键词列表
color_keywords = ["red", "green", "blue", "yellow", "purple", "black",
                  "white", "orange", "pink", "brown", "gray", "cyan"]


def extract_main_features(text, nlp):
    # 提取文本的主要特征，包括关键词、短语、实体等。
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.text.lower() not in nlp.Defaults.stop_words and token.pos_ != "PUNCT":
            lemma = token.lemma_
            keywords.append(lemma)
    print(list(set(keywords)))
    return list(set(keywords))


# 使用颜色空间比较计算颜色相似度
def calculate_color_similarity(color1, color2):
    # 定义颜色在HSV空间中的大致范围（以简化的方式）
    color_ranges = {
        "red": [(0, 100, 100), (10, 255, 255)],
        "green": [(50, 100, 100), (70, 255, 255)],
        "blue": [(110, 100, 100), (130, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        "purple": [(140, 100, 100), (160, 255, 255)],
        "black": [(0, 0, 0), (180, 255, 30)],
        "white": [(0, 0, 200), (180, 25, 255)],
        "orange": [(10, 100, 100), (20, 255, 255)],
        "pink": [(160, 100, 100), (180, 255, 255)],
        "brown": [(10, 50, 50), (20, 255, 200)],
        "gray": [(0, 0, 50), (180, 50, 200)],
        "cyan": [(80, 100, 100), (100, 255, 255)]
    }

    # 获取颜色范围的中心值作为代表
    def get_center_hsv(color):
        lower, upper = color_ranges[color]
        center_h = (lower[0] + upper[0]) / 2
        center_s = (lower[1] + upper[1]) / 2
        center_v = (lower[2] + upper[2]) / 2
        return center_h, center_s, center_v

    hsv1 = get_center_hsv(color1)
    hsv2 = get_center_hsv(color2)

    # 计算HSV空间中的欧氏距离
    distance = np.sqrt((hsv1[0] - hsv2[0]) ** 2 + (hsv1[1] - hsv2[1]) ** 2 + (hsv1[2] - hsv2[2]) ** 2)
    threshold = 100  # 阈值，可根据实际需求调整
    if distance > threshold:
        return 0
    # 转换为相似度分数（距离越小，相似度越高）
    similarity = 1 / (1 + distance)
    return similarity


# 修改后的相似度评分函数
def get_text_similarity(text1, text2, model, nlp):
    # 提取主要特征
    features1 = extract_main_features(text1, nlp)
    features2 = extract_main_features(text2, nlp)

    # 计算主要特征的匹配度
    main_similarity = 0.0
    for feature in features1:
        if feature in features2:
            main_similarity += 1.0  # 完全匹配
        else:
            for f in features2:
                # 检查是否为颜色相关特征并计算相似度
                if feature in color_keywords and f in color_keywords:
                    color_sim = calculate_color_similarity(feature, f)
                    main_similarity += color_sim
                else:
                    main_similarity += 0.1

    # 归一化主要特征相似度
    if len(features1) > 0:
        main_similarity /= len(features1)
    else:
        main_similarity = 0.0

    # 计算整体语义相似度
    embeddings = model.encode([text1, text2])
    semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # 结合主要特征相似度和语义相似度
    final_similarity = 0.7 * main_similarity + 0.3 * semantic_similarity
    return final_similarity


# 加载YOLOv8模型
def load_yolov8_model(model_path):
    print('正在加载YOLOv8模型...')
    model = YOLO(model_path)
    return model


# 加载ViT-GPT2模型
def load_vit_gpt2_model():
    print('正在加载ViT-GPT2模型...')
    vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_gpt2_model.to(device)
    return vit_gpt2_model, feature_extractor, tokenizer


# 加载Sentence Transformer模型
def load_sentence_transformer():
    print('正在加载Sentence Transformer模型...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


# 加载MarianMTModel和MarianTokenizer模型
def load_translation_model():
    print('正在加载MarianMTModel和MarianTokenizer模型...')
    translator_name = 'Helsinki-NLP/opus-mt-zh-en'
    translator_tokenizer = MarianTokenizer.from_pretrained(translator_name)
    translator_model = MarianMTModel.from_pretrained(translator_name)
    return translator_model, translator_tokenizer


# 翻译文本
def translate_text(text, translator_model, translator_tokenizer):
    # 确保分词器有 pad_token
    if translator_tokenizer.pad_token is None:
        translator_tokenizer.pad_token = translator_tokenizer.eos_token
    inputs = translator_tokenizer(text, padding="longest", return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output_ids = translator_model.generate(input_ids=input_ids, attention_mask=attention_mask)
    translated_text = translator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text


# 使用YOLOv8分割图像
def instance_segmentation(yolov8_model, image_path):
    image = cv2.imread(image_path)
    results = yolov8_model(image)
    instances = results[0].boxes.xywh  # 获取所有检测结果
    return instances, image


# 从分割结果中提取实例
def extract_instances(instances, image):
    cropped_images = []
    for i in range(len(instances)):
        # 获取边界框坐标
        box = instances[i].cpu().numpy()
        print(box)
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 裁剪图片
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)

    return cropped_images


# 使用OpenCV进行颜色识别并返回占比最多的三种颜色
def detect_colors(image):
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围和名称
    color_ranges = [
        (np.array([0, 100, 100]), np.array([10, 255, 255]), "red"),
        (np.array([50, 100, 100]), np.array([70, 255, 255]), "green"),
        (np.array([110, 100, 100]), np.array([130, 255, 255]), "blue"),
        (np.array([20, 100, 100]), np.array([30, 255, 255]), "yellow"),
        (np.array([140, 100, 100]), np.array([160, 255, 255]), "purple"),
        (np.array([0, 0, 0]), np.array([180, 255, 30]), "black"),
        (np.array([0, 0, 200]), np.array([180, 25, 255]), "white"),
        (np.array([10, 100, 100]), np.array([20, 255, 255]), "orange"),
        (np.array([160, 100, 100]), np.array([180, 255, 255]), "pink"),
        (np.array([10, 50, 50]), np.array([20, 255, 200]), "brown"),
        (np.array([0, 0, 50]), np.array([180, 50, 200]), "gray"),
        (np.array([80, 100, 100]), np.array([100, 255, 255]), "cyan")
    ]

    # 初始化颜色面积字典
    color_areas = {color_name: 0 for _, _, color_name in color_ranges}

    # 计算每种颜色的掩膜并累加面积
    for lower, upper, color_name in color_ranges:
        mask = cv2.inRange(hsv_image, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            color_areas[color_name] += area

    # 按面积从大到小排序颜色
    sorted_colors = sorted(color_areas.items(), key=lambda x: x[1], reverse=True)

    # 提取占比最多的三种颜色
    top_three_colors = [color[0] for color in sorted_colors[:3] if color[1] > 0]

    return top_three_colors


# 使用ViT-GPT2生成文本描述
def vit_gpt2_image_to_text(model, feature_extractor, tokenizer, instance_image):
    # 将OpenCV图像格式转换为PIL Image
    image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # 预处理图像
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")

    # 调整生成参数，增加生成长度和光束搜索数量
    max_length = 30  # 增加生成描述的最大长度
    num_beams = 6  # 增加光束搜索的数量

    # 生成文本
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 检查生成的描述中是否包含颜色信息
    contains_color = any(word in generated_text.lower() for word in color_keywords)

    if not contains_color:
        # 如果生成的描述中不包含颜色信息，使用OpenCV进行颜色检测
        colors = detect_colors(instance_image)
        if colors:
            # 将颜色信息整合到生成的描述中
            color_info = ", ".join(colors)
            generated_text = f"{generated_text} and the main colors are {color_info}."

    return generated_text


# 主函数
def main(input_image_path):
    # 初始化模型
    nlp = spacy.load("en_core_web_sm")
    yolov8_model = load_yolov8_model(r"D:\pycharm\python项目\PyTorch\Runs\detect\train\weights\best.pt")
    vit_gpt2_model, feature_extractor, tokenizer = load_vit_gpt2_model()
    sentence_transformer = load_sentence_transformer()
    translator_model, translator_tokenizer = load_translation_model()

    # 输入图像路径
    instances, image = instance_segmentation(yolov8_model, input_image_path)
    instances_list = extract_instances(instances, image)
    print(f"查询到{len(instances_list)}个实例")

    # 用户输入查询文本
    query_text = input('请输入要匹配的文本：\n')

    # 翻译用户查询文本为英文
    english_query_text = translate_text(query_text, translator_model, translator_tokenizer).strip('.')
    print('翻译后的文本：', english_query_text)

    # 保存匹配的实例和图像
    matched_instances = []
    output_dir = "matched_instances"
    os.makedirs(output_dir, exist_ok=True)

    max_score = 0.5
    best_instance_image = None
    for i, instance_image in enumerate(instances_list):
        # 使用ViT-GPT2转换为文本
        text_description = vit_gpt2_image_to_text(vit_gpt2_model, feature_extractor, tokenizer, instance_image)
        print(f"实例 {i+1} 图片描述: {text_description}")

        # 计算文本相似度
        similarity = get_text_similarity(english_query_text, text_description, sentence_transformer, nlp)
        print(f"相似度: {similarity:.2f}")

        # 如果相似度高于阈值，保存实例图像
        if similarity >= max_score:
            max_score = similarity
            best_instance_image = instance_image

    # 保存裁剪后的实例图像
    if best_instance_image is not None:
        cv2.imwrite(os.path.join(output_dir, "{}_instances.png".format(english_query_text)), best_instance_image)
        print("最高相似度的图片已保存到: {}".format(os.path.join(output_dir, "{}_instances.png".format(english_query_text))))
    else:
        print("没有匹配的图像")


# 运行主函数
if __name__ == "__main__":
    input_image_path = './Images/test3.jpg'
    main(input_image_path)