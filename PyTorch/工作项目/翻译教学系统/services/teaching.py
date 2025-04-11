from database.models import UserModel
from models.translator import TranslationModel
from models.knowledge import KnowledgeExtractor
import requests
import json
from typing import Dict, List, Any, Optional
from config import CONFIG


class DeepSeekTranslator:
    """使用DeepSeek API进行翻译和教学辅助"""

    def __init__(self):
        self.api_key = CONFIG['DEEPSEEK_API_KEY']
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def translate(self, chinese_text: str) -> str:
        """将中文翻译为英文"""
        prompt = f"""
        请将以下中文翻译成地道、自然的英文:

        {chinese_text}

        直接返回翻译结果，不要添加解释或其他内容。
        """

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"DeepSeek翻译API调用失败: {response.status_code}, {response.text}")
                return ""
        except Exception as e:
            print(f"DeepSeek翻译请求异常: {str(e)}")
            return ""



    def _fallback_get_initial_letters(self, english_text: str, word_count: int = 0) -> str:
        """API调用失败时的备用方法"""
        words = english_text.split()
        result = []

        for i, word in enumerate(words):
            if word_count == 0 or i < word_count:
                if word and word[0].isalpha():
                    # 保留首字母，其他替换为下划线
                    result.append(word[0] + '_' * (len(word) - 1))
                else:
                    # 对于标点符号等，保持原样
                    result.append(word)
            else:
                # 超出提示单词数量限制，全部替换为下划线
                result.append('_' * len(word))

        return ' '.join(result)

    def generate_layered_translation(self, english_text: str) -> List[str]:
        """生成分层翻译提示"""
        prompt = f"""
        请将以下英文句子分解成3-4个由简到难的层级，以便用于逐层学习:

        {english_text}

        返回一个JSON数组，包含从最简单到完整句子的各个层级。格式:
        ["简单主干", "基本扩展", "更多细节", "完整句子"]

        确保每个层级都是语法正确的句子，逐步增加复杂度。
        """

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"}
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    # 尝试解析JSON响应
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "layers" in data:
                        return data["layers"]
                    else:
                        # 返回格式不符合预期，使用原句作为唯一层级
                        return [english_text]
                except:
                    # JSON解析失败，返回原句
                    return [english_text]
            else:
                print(f"DeepSeek分层翻译API调用失败: {response.status_code}")
                return [english_text]
        except Exception as e:
            print(f"DeepSeek分层翻译请求异常: {str(e)}")
            return [english_text]


class TeachingService:
    def __init__(self):
        self.translator = TranslationModel()
        self.deepseek_translator = DeepSeekTranslator()
        self.knowledge_extractor = KnowledgeExtractor()
        self.user_model = UserModel()

    def layered_translation(self, chinese_text, layer_index=0):
        """逐层翻译模式

        参数:
            chinese_text (str): 中文原文
            layer_index (int): 层级索引，表示要返回的层级数
        """
        # 首先使用原有的翻译模型
        english_text = self.translator.translate(chinese_text)

        # 如果原有翻译模型返回空，则尝试使用DeepSeek
        if not english_text:
            english_text = self.deepseek_translator.translate(chinese_text)

        # 获取所有层级的分解，使用DeepSeek
        all_layers = self.deepseek_translator.generate_layered_translation(english_text)

        # 如果层级索引超出范围，则返回所有层级
        layers_to_return = all_layers[:layer_index + 1] if layer_index < len(all_layers) else all_layers

        return {
            'source': chinese_text,
            'reference': english_text,
            'layers': layers_to_return
        }

    def full_sentence_translation(self, chinese_text):
        """整句翻译模式"""
        # 首先使用原有的翻译模型
        english_text = self.translator.translate(chinese_text)

        # 如果原有翻译模型返回空，则尝试使用DeepSeek
        if not english_text:
            english_text = self.deepseek_translator.translate(chinese_text)

        return {
            'source': chinese_text,
            'reference': english_text
        }

    def get_knowledge_points(self, sentence):
        """获取知识点提示"""
        # 使用修改后的KnowledgeExtractor，它现在使用DeepSeek API
        result = self.knowledge_extractor.extract_knowledge(sentence)

        return {
            'grammar_points': result.get('grammar_points', []),
            'phrases': result.get('phrases', []),
            'collocations': result.get('collocations', [])
        }

    def record_translation_exercise(self, user_id, source_text, target_text, user_translation, accuracy,
                                    translation_type):
        """记录翻译练习"""
        return self.user_model.add_translation_record(user_id, source_text, target_text, user_translation, accuracy,
                                                      translation_type)