import os
import cv2
import torch
import spacy
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer, ViTImageProcessor
from transformers import VisionEncoderDecoderModel, AutoTokenizer

# 定义颜色关键词列表
color_keywords = ["red", "green", "blue", "yellow", "purple", "black",
                  "white", "orange", "pink", "brown", "gray", "cyan"]

car_categories = [
    "sedan", "hatchback", "suv", "coupe", "convertible",
    "wagon", "minivan", "pickup truck", "van", "truck",
    "bus", "motorbike", "scooter", "bicycle", "tricycle",
    "atv", "rv", "trailer", "fire engine", "ambulance",
    "police car", "taxi", "limousine", "electric car", "car"
]

people_categories = [
    "man", "woman", "child", "baby", "teenager", "elderly person",
    "doctor", "teacher", "engineer", "nurse", "police officer", "firefighter",
    "student", "parent", "friend", "neighbor", "customer", "passenger",
    "driver", "boss", "employee", "athlete", "artist", "musician",
    "scientist", "programmer", "lawyer", "judge", "cook", "chef", "human",
    "waiter", "waitress", "farmer", "soldier", "pilot", "astronaut", 'person', 'people'
]

# 初始化Flask应用
app: Flask = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 加载模型
print("正在加载模型中...")
nlp = spacy.load("en_core_web_sm")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
translator_name = 'Helsinki-NLP/opus-mt-zh-en'
translator_tokenizer = MarianTokenizer.from_pretrained(translator_name)
translator_model = MarianMTModel.from_pretrained(translator_name)
vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_gpt2_model.to(device)
yolov8_model = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\detect\train\weights\best.pt")
print("全部加载完成！")


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
def calculate_color_similarity(color1, color2, ratio):
    # 定义颜色在HSV空间中的大致范围（以简化的方式）
    color_ranges = {
        "red": [(0, 50, 50), (10, 255, 255)],  # 红色
        "green": [(35, 50, 50), (85, 255, 255)],  # 绿色
        "blue": [(100, 50, 50), (140, 255, 255)],  # 蓝色
        "yellow": [(15, 50, 50), (35, 255, 255)],  # 黄色
        "purple": [(140, 50, 50), (165, 255, 255)],  # 紫色
        "black": [(0, 0, 0), (180, 255, 30)],  # 黑色
        "white": [(0, 0, 200), (180, 25, 255)],  # 白色
        "orange": [(10, 50, 50), (20, 255, 255)],  # 橙色
        "pink": [(160, 50, 50), (180, 255, 255)],  # 粉色
        "brown": [(10, 50, 50), (20, 255, 200)],  # 棕色
        "gray": [(0, 0, 50), (180, 50, 200)],  # 灰色
        "cyan": [(80, 50, 50), (100, 255, 255)]  # 青色
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
    similarity = ratio / (1 + distance)
    return similarity


# 修改后的相似度评分函数
def get_text_similarity(text1, text2, colors, model, nlp):
    # 提取主要特征
    features1 = extract_main_features(text1, nlp)
    features2 = extract_main_features(text2, nlp)

    # 计算主要特征的匹配度
    main_similarity = 0.0
    for feature in features1:
        if feature in features2:
            main_similarity += 1
            # 检查是否为颜色相关特征并计算相似度
        else:
            for f in features2:
                if feature in color_keywords and f in color_keywords:
                    ratio = 1 if not colors else colors[f]
                    color_sim = calculate_color_similarity(feature, f, ratio)
                    main_similarity += color_sim
                if feature.lower() in car_categories and f.lower() in car_categories:
                    main_similarity += 1
                if feature.lower() in people_categories and f.lower() in people_categories:
                    main_similarity += 1

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
    print(final_similarity)
    return final_similarity


def translate_text(text, translator_model, translator_tokenizer):
    if translator_tokenizer.pad_token is None:
        translator_tokenizer.pad_token = translator_tokenizer.eos_token
    inputs = translator_tokenizer(text, padding="longest", return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output_ids = translator_model.generate(input_ids=input_ids, attention_mask=attention_mask)
    translated_text = translator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text


def instance_segmentation(yolov8_model, image_path):
    image = cv2.imread(image_path)
    results = yolov8_model(image)
    instances = results[0].boxes.xywh
    classes = results[0].boxes.cls.tolist()  # 获取每个实例的类别
    return instances, image, classes  # 返回实例、图像和类别


def extract_instances(instances, image, classes):
    cropped_images = []
    for i in range(len(instances)):
        box = instances[i].cpu().numpy()
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append([cropped_image, classes[i]])  # 将切割的图像和类别一起返回
    return cropped_images


def extract_colors(instance_image):
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2HSV)

    # 使用K-means聚类提取主色调，增加聚类数以捕捉更多颜色细节
    pixels = hsv_image.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=10, n_init=10)  # 增加聚类数
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 计算每种颜色的占比
    color_ratios = {}
    for i in range(len(centers)):
        color_ratios[i] = np.sum(labels == i) / len(labels)

    color_ranges = [
        ("red", [0, 50, 50], [10, 255, 255]),
        ("green", [35, 50, 50], [85, 255, 255]),
        ("blue", [100, 50, 50], [140, 255, 255]),
        ("yellow", [15, 50, 50], [35, 255, 255]),
        ("purple", [140, 50, 50], [165, 255, 255]),
        ("black", [0, 0, 0], [180, 255, 30]),
        ("white", [0, 0, 200], [180, 25, 255]),
        ("orange", [10, 50, 50], [20, 255, 255]),
        ("pink", [160, 50, 50], [180, 255, 255]),
        ("brown", [10, 50, 50], [20, 255, 200]),
        ("gray", [0, 0, 50], [180, 50, 200]),
        ("cyan", [80, 50, 50], [100, 255, 255])
    ]

    # 计算每种聚类中心颜色与预定义颜色的相似度
    color_matches = {}
    for i, center in enumerate(centers):
        min_distance = float('inf')
        matched_color = "unknown"
        for name, lower, upper in color_ranges:
            # 计算中心颜色与预定义颜色范围的中心点的距离
            center_hsv = center
            range_center = np.array([(lower[0] + upper[0])/2, (lower[1] + upper[1])/2, (lower[2] + upper[2])/2])
            dist = distance.euclidean(center_hsv, range_center)
            if dist < min_distance:
                min_distance = dist
                matched_color = name
        # 根据颜色占比和距离计算权重
        weight = color_ratios[i] * (1 / (1 + min_distance))
        if matched_color in color_matches:
            color_matches[matched_color] += weight
        else:
            color_matches[matched_color] = weight

    # 按权重从高到低排序颜色
    sorted_colors = sorted(color_matches.items(), key=lambda x: x[1], reverse=True)

    # 提取权重最高的三种颜色及其权重
    colors = {}
    for color, ratio in sorted_colors[:3]:
        colors[color] = ratio

    return colors


# 使用颜色识别和yolo类别识别生成伪图片描述
def generate_pseudo_description(instance_image, class_id):
    classes = ['person', 'car']
    # 使用SAM和颜色识别函数检测颜色
    colors = extract_colors(instance_image)
    color_info = ", ".join(colors.keys()) if colors else "unknown"

    # 生成伪图片描述
    pseudo_description = f"a {color_info} {classes[int(class_id)]}"
    return pseudo_description, colors


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file:
        # 生成唯一文件名
        import uuid
        unique_filename = str(uuid.uuid4()) + '.jpg'
        image_path = "static/uploads/" + unique_filename
        file.save(image_path)
        print(image_path)
        return jsonify({'message': 'Image uploaded successfully', 'image_path': image_path})


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query_text')
    image_path = data.get('image_path')
    image_path = "D:/pycharm/python项目/PyTorch/工作项目/跨模态物体检测/web/" + str(image_path)

    # 翻译用户查询文本为英文
    english_query_text = translate_text(query_text, translator_model, translator_tokenizer).strip('.')
    file_name = english_query_text + '.jpg'

    # 检测和分割图像
    instances, image, classes = instance_segmentation(yolov8_model, image_path)
    instances_list = extract_instances(instances, image, classes)

    # 查找最佳匹配实例
    index = 0
    max_score = 0.5
    best_instance_image = None
    for i, instance_image in enumerate(instances_list):
        pseudo_description, colors = generate_pseudo_description(instance_image[0], instance_image[1])
        print(pseudo_description)
        similarity = get_text_similarity(english_query_text, pseudo_description, colors, sentence_transformer, nlp)
        if similarity >= max_score:
            max_score = similarity
            index = i
            best_instance_image = instance_image[0]

    # 保存最佳匹配实例
    if best_instance_image is not None:
        # 获取匹配实例的边界框
        box = instances[index].cpu().numpy()
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 在原图中绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 保存处理后的图片
        processed_image_path = "D:/pycharm/python项目/PyTorch/工作项目/跨模态物体检测/web/static/results/" + file_name
        cv2.imwrite(processed_image_path, image)
        print(processed_image_path)

        return jsonify({'message': 'Search completed', 'result_path': '/static/results/' + file_name})
    else:
        return jsonify({'message': 'No matching image found'})


if __name__ == '__main__':
    app.run(debug=True)