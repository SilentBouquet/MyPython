import re
import sys
import cv2
import jieba
import torch
from PIL import Image
from cnocr import CnOcr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from PyQt5.QtCore import Qt
import language_tool_python
from snownlp import SnowNLP
from PyQt5.QtGui import QPixmap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import cosine_similarity as torch_cosine_similarity
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QFileDialog


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
        student_embedding = self.model.encode([student_text], device="cuda")      # 使用 GPU

        # 相似性计算
        cosine_sim = torch_cosine_similarity(torch.tensor(reference_embedding), torch.tensor(student_embedding)).item()

        student_errors = self.tool.check(student_text)
        grammar_score = 1 - (len(student_errors) / len(student_text.split())) if len(student_text.split()) != 0 else 0

        innovation_score = self.assess_innovation(reference_text, student_text)
        emotional_score = self.assess_emotional_expression(student_text)

        content_score = cosine_sim * 50 * self.p
        grammar_score = grammar_score * 10 * self.p
        innovation_score = innovation_score * 0.2 * self.p
        emotional_score = emotional_score * 0.2 * self.p

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
def extract_text(img, ocr_model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = ocr_model.ocr(img_pil)
    extracted_text = ''.join([item['text'] for item in result])
    extracted_text = re.sub(r'[^\w\s]', '', extracted_text)  # 移除所有标点
    return extracted_text


def extract_trocr_text(img, processor, model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to("cuda")  # 使用 GPU
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = re.sub(r'[^\w\s]', '', generated_text)  # 移除所有标点
    return generated_text


def extract_trocr_text(img, ocr_model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = ocr_model.ocr(img_pil)
    extracted_text = ''.join([item['text'] for item in result])
    extracted_text = re.sub(r'[^\w\s]', '', extracted_text)  # 移除所有标点
    for i in ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D']:
        if i in extracted_text:
            extracted_text = i.upper()
    return extracted_text


class InteractiveInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_path = ''
        self.init_models()  # 初始化预加载的模型

    def init_models(self):
        # 预加载模型，避免重复加载
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(
            "cuda")  # 使用 GPU
        self.cnocr_model = CnOcr(rec_model_name='densenet_lite_136-gru')  # 使用 CNOCR 模型
        self.xuanze = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\xuanze\train\weights\best.pt")
        self.jieda = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\jieda\train\weights\best.pt")
        self.tiankong = YOLO(r"D:\pycharm\python项目\PyTorch\Runs\tiankong\train\weights\best.pt")

    def init_ui(self):
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle("自动作业批改系统")

        main_layout = QHBoxLayout()

        # 左侧布局
        left_widget = QWidget()
        left_widget.setFixedWidth(500)
        left_widget.setStyleSheet("background-color: #f0f0f0;")
        left_layout = QVBoxLayout(left_widget)

        self.left_label = QLabel("点击上传照片")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
        left_layout.addWidget(self.left_label)

        left_button = QPushButton("上传照片")
        left_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005CA9;
            }
            QPushButton:pressed {
                background-color: #003366;
            }
        """)
        left_button.setCursor(Qt.PointingHandCursor)
        left_button.clicked.connect(self.upload_photo)
        left_layout.addWidget(left_button)

        # 右侧布局
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        upper_right_layout = QVBoxLayout()

        # 选择题型
        type_container = QWidget()
        type_container.setStyleSheet("background-color: white;")
        type_layout = QHBoxLayout(type_container)
        type_label = QLabel("选择题型：")
        type_label.setFixedWidth(80)
        type_label.setStyleSheet("font-size: 16px;")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["选择题", "填空题", "解答题", "简答题", "作文题"])
        self.type_combo.setStyleSheet("""
                QComboBox {
                    height: 30px;
                    background-color: white;
                    border: 1px solid #ccc;
                    padding: 5px;
                    font-size: 16px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 25px;
                    border-left-width: 1px;
                    border-left-color: #ccc;
                    border-left-style: solid;
                    border-top-right-radius: 3px;
                    border-bottom-right-radius: 3px;
                }
                QComboBox QAbstractItemView {
                    selection-background-color: #f0f0f0;
                    background-color: white;
                    border: 1px solid #ccc;
                    selection-color: #000000;
                    font-size: 16px;
                    min-height: 30px;
                }
            """)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        upper_right_layout.addWidget(type_container)

        # 参考答案
        answer_widget = QWidget()
        answer_widget.setStyleSheet("background-color: white;")
        answer_layout = QHBoxLayout(answer_widget)
        answer_label = QLabel("参考答案：")
        answer_label.setFixedWidth(80)
        answer_label.setStyleSheet("font-size: 16px;")
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("请输入参考答案")
        self.answer_input.setStyleSheet("""
            QLineEdit {
                height: 25px;
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                font-size: 16px;
            }
        """)

        answer_layout.addWidget(answer_label)
        answer_layout.addWidget(self.answer_input)
        upper_right_layout.addWidget(answer_widget)

        # 分值
        score_widget = QWidget()
        score_widget.setStyleSheet("background-color: white;")
        score_layout = QHBoxLayout(score_widget)
        score_label = QLabel("分值：")
        score_label.setFixedWidth(80)
        score_label.setStyleSheet("font-size: 16px;")
        self.score_input = QLineEdit()
        self.score_input.setPlaceholderText("请输入题目分值")
        self.score_input.setStyleSheet("""
            QLineEdit {
                height: 25px;
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                font-size: 16px;
            }
        """)

        score_layout.addWidget(score_label)
        score_layout.addWidget(self.score_input)
        upper_right_layout.addWidget(score_widget)

        # 计算按钮
        self.calculate_button = QPushButton("计算得分")
        self.calculate_button.setStyleSheet("""
            QPushButton {
                height: 20px;
                background-color: #007BFF;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #005CA9;
            }
            QPushButton:pressed {
                background-color: #003366;
            }
        """)
        self.calculate_button.setCursor(Qt.PointingHandCursor)
        self.calculate_button.clicked.connect(self.calculate_score)
        upper_right_layout.addWidget(self.calculate_button)

        # 结果显示
        lower_right_layout = QVBoxLayout()
        result_label = QLabel("得分和建议：")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFontPointSize(14)
        self.result_text.setStyleSheet("border: 1px solid #ccc; padding: 10px; font-size: 16px;")

        lower_right_layout.addWidget(result_label)
        lower_right_layout.addWidget(self.result_text)

        right_layout.addLayout(upper_right_layout)
        right_layout.addLayout(lower_right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: white;")

    def upload_photo(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "上传照片", "", "图片文件 (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(500, 300, Qt.KeepAspectRatio)
            self.left_label.setPixmap(scaled_pixmap)
            self.left_label.setAlignment(Qt.AlignCenter)

    def calculate_score(self):
        result = ""
        self.calculate_button.setEnabled(False)

        question_type = self.type_combo.currentText()
        reference_answer = self.answer_input.text().strip()
        total_score = self.score_input.text().strip()

        if not reference_answer or not total_score or not question_type:
            self.calculate_button.setEnabled(True)
            result += "请确保上传了照片、输入了参考答案和分值！\n"
            self.result_text.setText(result)
            return
        cls = 0
        self.student_answer = ""
        total_score = eval(total_score)

        if question_type == "选择题":
            split_answer = SplitAnswer(self.xuanze)
            cls, cropped_image_list = split_answer.crop_answers(self.file_path)
            result += f'识别类别为{cls}\n'
            for img in cropped_image_list:
                extracted_text = extract_trocr_text(img, self.trocr_processor, self.trocr_model)
                if extracted_text:
                    self.student_answer += extracted_text
            idx = [i.start() for i in re.finditer(r'[A-D]', self.student_answer.upper())]
            if idx:
                self.student_answer = self.student_answer.upper()[idx[0]]
            else:
                self.student_answer = ''
            if not self.student_answer:
                self.calculate_button.setEnabled(True)
                result += "未识别到学生答案\n"
                self.result_text.setText(result)
                return
            result += f"学生答案为：{self.student_answer}\n"
            if self.student_answer.strip() == reference_answer.strip():
                score = total_score
                suggestion = "答案正确！"
            else:
                score = 0
                suggestion = "答案错误，请检查！"
            result += f"得分：{score}/{total_score}\n参考建议：{suggestion}\n"
        elif question_type == "填空题":
            split_answer = SplitAnswer(self.tiankong)
            cls, cropped_image_list = split_answer.crop_answers(self.file_path)
            result += f'识别类别为{cls}\n'
            for img in cropped_image_list:
                extracted_text = extract_text(img, self.cnocr_model)
                if extracted_text:
                    self.student_answer += extracted_text + '\n'
            if not self.student_answer:
                self.calculate_button.setEnabled(True)
                result += "未识别到学生答案\n"
                self.result_text.setText(result)
                return
            result += f"学生答案为：{self.student_answer}\n"
            if self.student_answer.strip() == reference_answer.strip():
                score = total_score
                suggestion = "答案正确！"
            else:
                score = 0
                suggestion = "答案错误，请检查！"
            result += f"得分：{score}/{total_score}\n参考建议：{suggestion}\n"
        elif question_type == "解答题":
            split_answer = SplitAnswer(self.jieda)
            cls, cropped_image_list = split_answer.crop_answers(self.file_path)
            result += f'识别类别为{cls}\n'
            for img in cropped_image_list:
                extracted_text = extract_text(img, self.cnocr_model)
                if extracted_text:
                    self.student_answer += extracted_text + '\n'
            if not self.student_answer:
                self.calculate_button.setEnabled(True)
                result += "未识别到学生答案\n"
                self.result_text.setText(result)
                return
            result += f"学生答案为：{self.student_answer}\n"
            score, suggestion = grade_math_physics_answer(self.student_answer, reference_answer, total_score)
            result += f"得分：{score}/{total_score}\n参考建议：{suggestion}"
        elif question_type == "作文题":
            img = cv2.imread(self.file_path)
            extracted_text = extract_text(img, self.cnocr_model)
            self.student_answer += extracted_text
            if not self.student_answer:
                self.calculate_button.setEnabled(True)
                result += "未识别到学生答案\n"
                self.result_text.setText(result)
                return
            result += f"学生答案为：{self.student_answer}\n"
            OTQ = OpenTypeQuestions(total_score, student_text=self.student_answer, reference_text=reference_answer)
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

        self.result_text.setText(result)
        self.calculate_button.setEnabled(True)
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveInterface()
    window.show()
    sys.exit(app.exec_())