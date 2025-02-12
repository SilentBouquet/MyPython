import os
import cv2
import spacy
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from ultralytics import YOLO

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


# 加载MarianMTModel和MarianTokenizer模型
def load_translation_model():
    print('正在加载MarianMTModel和MarianTokenizer模型...')
    return translator_model, translator_tokenizer


def extract_main_features(text, nlp):
    # 提取文本的主要特征，包括关键词、短语、实体等。
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.text.lower() not in nlp.Defaults.stop_words and token.pos_ != "PUNCT":
            lemma = token.lemma_
            keywords.append(lemma)
    return list(set(keywords))


def get_text_similarity(text1, text2, model, nlp):
    # 提取主要特征
    features1 = extract_main_features(text1, nlp)
    features2 = extract_main_features(text2, nlp)
    main_similarity = 0.0
    for feature in features1:
        if feature in features2:
            main_similarity += 1.0
        else:
            main_similarity += 0.1
    if len(features1) > 0:
        main_similarity /= len(features1)
    else:
        main_similarity = 0.0
    embeddings = model.encode([text1, text2])
    semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    final_similarity = 0.7 * main_similarity + 0.3 * semantic_similarity
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
    return instances, image


def extract_instances(instances, image):
    cropped_images = []
    for i in range(len(instances)):
        box = instances[i].cpu().numpy()
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
    return cropped_images


def vit_gpt2_image_to_text(model, feature_extractor, tokenizer, instance_image):
    image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return generated_text


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
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(image_path)
        return jsonify({'message': 'Image uploaded successfully', 'image_path': image_path})


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query_text')
    image_path = data.get('image_path')

    # 翻译用户查询文本为英文
    english_query_text = translate_text(query_text, translator_model, translator_tokenizer).strip('.')

    image_path = "D:/pycharm/python项目/PyTorch/工作项目/跨模态物体检测/web" + str(image_path)

    # 检测和分割图像
    instances, image = instance_segmentation(yolov8_model, image_path)
    instances_list = extract_instances(instances, image)

    # 查找最佳匹配实例
    max_score = 0.6
    index = 0
    best_instance_image = None
    for i, instance_image in enumerate(instances_list):
        text_description = vit_gpt2_image_to_text(vit_gpt2_model, feature_extractor, tokenizer, instance_image)
        similarity = get_text_similarity(english_query_text, text_description, sentence_transformer, nlp)
        if similarity >= max_score:
            index = i
            max_score = similarity
            best_instance_image = instance_image

    # 在原图中框出匹配对象
    if best_instance_image is not None:
        # 获取匹配实例的边界框
        box = instances[index].cpu().numpy()
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 在原图中绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 保存处理后的图片
        processed_image_path = "D:/pycharm/python项目/PyTorch/工作项目/跨模态物体检测/vertical/static/results/result.jpg"
        cv2.imwrite(processed_image_path, image)

        return jsonify({'message': 'Search completed',  'result_path': '/static/results/result.jpg'})
    else:
        return jsonify({'message': 'No matching image found'})


if __name__ == '__main__':
    app.run(debug=True)