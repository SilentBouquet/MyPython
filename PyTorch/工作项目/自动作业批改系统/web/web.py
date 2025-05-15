import os
import re
import cv2
import jieba
import torch
from PIL import Image
from paddleocr import PaddleOCR
from ultralytics import YOLO
import language_tool_python
from snownlp import SnowNLP
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import cosine_similarity as torch_cosine_similarity
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# 初始化模型
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to("cuda")  # 使用 GPU
paddleocr_model = PaddleOCR(use_angle_cls=True, lang="ch")  # 使用 PaddleOCR
xuanze_model = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\xuanze\train\weights\best.pt")
jieda_model = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\jieda\train\weights\best.pt")
tiankong_model = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\tiankong\train\weights\best.pt")


# 作文评分系统
class OpenTypeQuestions(object):
    def __init__(self, p, reference_text, student_text):
        self.p = p
        self.reference_text = reference_text
        self.student_text = student_text
        self.tool = language_tool_python.LanguageTool('zh-CN')
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2').to("cuda")  # 使用 GPU

    def preprocess(self, text):
        words = jieba.cut(text)
        return " ".join(words)

    def assess_emotional_expression(self, text):
        s = SnowNLP(text)
        sentiment_score = s.sentiments
        emotional_words = len([word for word in text.split() if SnowNLP(word).sentiments > 0.5])
        emotional_density = emotional_words / len(text.split())
        return sentiment_score * 50 + emotional_density * 50

    def assess_innovation(self, reference, student):
        reference_words = set(reference.split())
        student_words = set(student.split())
        unique_words = student_words - reference_words
        innovation_score = len(unique_words) / len(student_words)
        return innovation_score * 100

    def open_type_question(self):
        reference_text = self.preprocess(self.reference_text)
        student_text = self.preprocess(self.student_text)

        reference_embedding = self.model.encode([reference_text], device="cuda")  # 使用 GPU
        student_embedding = self.model.encode([student_text], device="cuda")  # 使用 GPU

        # 相似性计算
        cosine_sim = torch_cosine_similarity(torch.tensor(reference_embedding), torch.tensor(student_embedding)).item()

        student_errors = self.tool.check(student_text)
        grammar_score = 1 - (len(student_errors) / len(student_text.split())) if len(student_text.split()) != 0 else 0

        innovation_score = self.assess_innovation(reference_text, student_text)
        emotional_score = self.assess_emotional_expression(student_text)

        content_score = cosine_sim * self.p * 0.4
        grammar_score = grammar_score * self.p * 0.2
        innovation_score = innovation_score * 0.2 * self.p / 100
        emotional_score = emotional_score * 0.2 * self.p / 100

        final_score = content_score + grammar_score + innovation_score + emotional_score

        return {
            "content": round(content_score, 2),
            "grammar": round(grammar_score, 2),
            "innovation": round(innovation_score, 2),
            "emotion": round(emotional_score, 2),
            "score": round(final_score, 2)
        }


# 切割答案与题目
class SplitAnswer(object):
    def __init__(self, model):
        self.model = model

    def load_model_and_predict(self, image_path, conf_threshold=0.5):
        results = self.model(image_path)
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) in result.names:
                    cls = int(box.cls[0])
                    conf = box.conf[0].item()
                    xyxy = box.xyxy[0].tolist()
                    if cls == 0 and conf >= conf_threshold:
                        predictions.append({
                            'class': cls,
                            'confidence': conf,
                            'box': xyxy
                        })
        return predictions

    def crop_answers(self, image_path):
        cls = 0
        cropped_image_list = []
        predictions = self.load_model_and_predict(image_path)
        if not predictions:
            return cls, cropped_image_list
        image = cv2.imread(image_path)
        if image is None:
            return cls, cropped_image_list
        for pred in predictions:
            cls = pred['class']
            box = pred['box']
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_list.append(cropped_image)
        return cls, cropped_image_list


def grade_math_physics_answer(student_answer, reference_answer, total_score):
    student_answer = re.sub(r'\s+', ' ', student_answer).strip()
    reference_answer = re.sub(r'\s+', ' ', reference_answer).strip()

    if not student_answer:
        return 0, "你的答案为空，请补充完整答案。"

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_answer, reference_answer])
    similarity_score = (tfidf_matrix * tfidf_matrix.T)[0, 1]

    score = round(similarity_score * total_score)

    if similarity_score >= 0.8:
        suggestion = "你的答案与参考答案高度相似，得满分！"
    elif similarity_score >= 0.5:
        suggestion = "你的答案部分正确，但存在一些问题，请检查并改进。"
    else:
        suggestion = "你的答案与参考答案差异较大，请重新检查解题过程和答案。"

    return score, suggestion


# 增加快速OCR处理
def extract_text_paddleocr(img, ocr_model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = ocr_model.ocr(img_pil, cls=True)
    extracted_text = ''.join([line[1][0] for line in result[0]])
    extracted_text = re.sub(r'[^\w\s]', '', extracted_text)  # 移除所有标点
    return extracted_text


def extract_trocr_text(img, processor, model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to("cuda")  # 使用 GPU
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = re.sub(r'[^\w\s]', '', generated_text)  # 移除所有标点
    return generated_text


# 在图片上添加文字
def add_text_to_image(image_path, text, output_path):
    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 设置文字参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # 红色
    thickness = 2

    # 获取文字尺寸
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算右下角位置
    img_height, img_width = img.shape[:2]
    bottom_right_corner_x = img_width - text_size[0] - 10  # 右下角x坐标
    bottom_right_corner_y = img_height - 10  # 右下角y坐标

    # 添加文字
    cv2.putText(img, text, (bottom_right_corner_x, bottom_right_corner_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # 保存图片
    cv2.imwrite(output_path, img)

    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})

    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        return jsonify({'message': '文件上传成功', 'file_path': file_path})


@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    data = request.json
    question_type = data.get('question_type')
    reference_answer = data.get('reference_answer').strip()
    total_score = float(data.get('total_score'))
    file_path = data.get('file_path')

    score = 0
    result = ""
    student_answer = ""

    if question_type == "选择题":
        split_answer = SplitAnswer(xuanze_model)
        cls, cropped_image_list = split_answer.crop_answers(file_path)
        for img in cropped_image_list:
            extracted_text = extract_trocr_text(img, trocr_processor, trocr_model)
            if extracted_text:
                student_answer += extracted_text
        idx = [i.start() for i in re.finditer(r'[A-D]', student_answer.upper())]
        if idx:
            student_answer = student_answer.upper()[idx[0]]
        else:
            student_answer = ''
        if not student_answer:
            return jsonify({'error': '未识别到学生答案'})
        result += f"学生答案为：{student_answer}\n"
        if student_answer.strip() == reference_answer.strip():
            score = total_score
            suggestion = "答案正确！"
        else:
            score = 0
            suggestion = "答案错误，请检查！"
        result += f"得分：{score}/{total_score}\n参考建议：{suggestion}\n"
    elif question_type == "填空题":
        split_answer = SplitAnswer(tiankong_model)
        cls, cropped_image_list = split_answer.crop_answers(file_path)
        for img in cropped_image_list:
            extracted_text = extract_text_paddleocr(img, paddleocr_model)  # 使用 PaddleOCR
            if extracted_text:
                student_answer += extracted_text + '\n'
        if not student_answer:
            return jsonify({'error': '未识别到学生答案'})
        result += f"学生答案为：{student_answer}\n"
        if student_answer.strip() == reference_answer.strip():
            score = total_score
            suggestion = "答案正确！"
        else:
            score = 0
            suggestion = "答案错误，请检查！"
        result += f"得分：{score}/{total_score}\n参考建议：{suggestion}"
    elif question_type == "解答题":
        split_answer = SplitAnswer(jieda_model)
        cls, cropped_image_list = split_answer.crop_answers(file_path)
        for img in cropped_image_list:
            extracted_text = extract_text_paddleocr(img, paddleocr_model)  # 使用 PaddleOCR
            if extracted_text:
                student_answer += extracted_text + '\n'
        if not student_answer:
            return jsonify({'error': '未识别到学生答案'})
        result += f"学生答案为：{student_answer}\n"
        score, suggestion = grade_math_physics_answer(student_answer, reference_answer, total_score)
        result += f"得分：{score}/{total_score}\n参考建议：{suggestion}"
    elif question_type == "作文题":
        img = cv2.imread(file_path)
        extracted_text = extract_text_paddleocr(img, paddleocr_model)  # 使用 PaddleOCR
        student_answer += extracted_text
        if not student_answer:
            return jsonify({'error': '未识别到学生答案'})
        result += f"学生答案为：{student_answer}\n"
        OTQ = OpenTypeQuestions(total_score, student_text=student_answer, reference_text=reference_answer)
        scores = OTQ.open_type_question()
        content_score = scores["content"]
        grammar_score = scores["grammar"]
        innovation_score = scores["innovation"]
        emotion_score = scores["emotion"]
        score = scores["score"]
        result += f"内容相关性得分：{content_score:.2f}\n"
        result += f"语法检测得分：{grammar_score:.2f}\n"
        result += f"创新性得分：{innovation_score:.2f}\n"
        result += f"情感表达得分：{emotion_score:.2f}\n"
        result += f"总分：{score:.2f}"

    # 保存处理后的图片路径
    processed_image_path = 'D:/pycharm/python项目/PyTorch/工作项目/自动作业批改系统/web/static/results/processed_' + os.path.basename(file_path)
    add_text_to_image(file_path, f"{score}", processed_image_path)

    # 返回结果和图片
    return jsonify({'result': result, 'image_path': '/static/results/processed_' + os.path.basename(file_path)})


if __name__ == '__main__':
    app.run(debug=True)