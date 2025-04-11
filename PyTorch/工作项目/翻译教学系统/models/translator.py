from transformers import MarianMTModel, MarianTokenizer
import torch
from config import CONFIG
import random
import os


class TranslationModel:
    def __init__(self):
        self.model_name = CONFIG['model']['name']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
        # 确保data文件夹存在
        os.makedirs('data', exist_ok=True)
        self.sentences_file = 'data/sentences.txt'

    def translate(self, text):
        """将中文翻译成英文"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def get_initial_letters(self, text, count=0):
        """获取目标句子中每个单词的首字母

        参数:
            text (str): 要处理的文本
            count (int): 要显示首字母的单词数量，0表示显示全部
        """
        words = text.split()
        result = []

        for word in words:
            if len(word) > 0:
                result.append(word[0] + '_' * (len(word) - 1))

        if count == 0:
            return ' '.join(result)
        else:
            return ' '.join(result[:count])

    def get_random_sentences(self):
        """从预定义的文件中随机抽取句子"""
        try:
            # 读取文件中的所有句子
            with open(self.sentences_file, 'r', encoding='utf-8') as file:
                # 读取所有行并去除首尾的换行符等空白字符
                lines = [line.strip() for line in file]
                # 过滤掉空字符串（即空行）
                all_sentences = [line for line in lines if line]

            # 随机选择一个句子
            selected = random.choice(all_sentences)
            print(selected)

            # 生成翻译和提示
            translation = self.translate(selected)
            print(translation)

            return {
                "source": selected,
                "reference": translation,
                "prompt": self.get_initial_letters(translation, 0)  # 默认显示每个单词的第一个字母
            }

        except Exception as e:
            print(f"读取句子文件出错: {e}")
            # 出错时返回一个默认句子
            default_sentence = "我们正在学习如何用英语表达自己的想法和观点。"
            translation = self.translate(default_sentence)
            return {
                "source": default_sentence,
                "reference": translation,
                "prompt": self.get_initial_letters(translation, 0)
            }

    def add_custom_sentence(self, chinese):
        """添加自定义句子到句子文件
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.sentences_file), exist_ok=True)

            # 将句子追加到文件末尾
            with open(self.sentences_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{chinese}\n")

            return True
        except Exception as e:
            print(f"添加自定义句子时出错: {str(e)}")
            raise