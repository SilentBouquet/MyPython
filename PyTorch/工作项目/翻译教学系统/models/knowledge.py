import requests
import json
import os
from typing import Dict, List, Any
from config import CONFIG
from database.models import UserModel


class KnowledgeExtractor:
    def __init__(self):
        self.api_key = CONFIG['DEEPSEEK_API_KEY']
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # 初始化数据库模型
        self.user_model = UserModel()

    def extract_knowledge(self, sentence: str) -> Dict[str, List[Dict[str, Any]]]:
        """使用DeepSeek API提取文本中的所有知识点"""
        try:
            # 调用DeepSeek API进行知识点提取
            deepseek_analysis = self._call_deepseek_api(sentence)

            if deepseek_analysis:
                # 将新知识点保存到数据库
                self._save_to_database(deepseek_analysis)
                return deepseek_analysis
            else:
                # 如果API调用失败，返回空结果
                return {
                    'grammar_points': [],
                    'phrases': [],
                    'collocations': []
                }
        except Exception as e:
            print(f"提取知识点时出错: {str(e)}")
            # 出错时返回空结果
            return {
                'grammar_points': [],
                'phrases': [],
                'collocations': []
            }

    def _call_deepseek_api(self, sentence: str) -> Dict[str, List[Dict[str, Any]]]:
        """调用DeepSeek API提取知识点"""
        prompt = f"""
        请分析以下英语句子，提取其中的语法点、短语和搭配知识。

        句子: {sentence}

        请以JSON格式返回分析结果，包括以下三类内容:

        1. 语法点(grammar_points): 提取句子中的语法结构，如主语、谓语、宾语、从句等
        2. 短语(phrases): 提取名词短语、动词短语等
        3. 词语搭配(collocations): 提取形容词+名词、动词+副词等常见搭配

        对于每个知识点，请提供:
        - text: 文本内容
        - explanation: 中文解释
        - difficulty: 难度级别(1=简单, 2=中等, 3=困难)
        - type: 类型(如nsubj, dobj, noun_phrase等)

        整体JSON格式应为:
        {{
            "grammar_points": [
                {{"text": "...", "explanation": "...", "difficulty": 数字, "type": "...", "id": "auto_generated"}}
            ],
            "phrases": [
                {{"text": "...", "explanation": "...", "difficulty": 数字, "type": "...", "id": "auto_generated"}}
            ],
            "collocations": [
                {{"text": "...", "explanation": "...", "difficulty": 数字, "type": "...", "id": "auto_generated"}}
            ]
        }}

        请确保分析全面且准确，并将相似内容去重。
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
                data = json.loads(content)

                # 为每个知识点添加ID
                self._add_ids_to_knowledge_points(data)

                return data
            else:
                print(f"DeepSeek API 调用失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"DeepSeek API 调用异常: {str(e)}")
            return None

    def _add_ids_to_knowledge_points(self, data):
        """为知识点添加唯一ID"""
        import hashlib

        for category in ['grammar_points', 'phrases', 'collocations']:
            if category in data:
                for item in data[category]:
                    # 使用内容生成唯一ID
                    text = item.get('text', '')
                    item_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    item['id'] = item_hash[:10]  # 使用前10位作为ID

    def _save_to_database(self, new_knowledge):
        """将新知识点保存到数据库"""
        try:
            # 添加每个类别的知识点
            for category in ['grammar_points', 'phrases', 'collocations']:
                for item in new_knowledge.get(category, []):
                    knowledge_id = item.get('id')
                    knowledge_type = item.get('type', '')
                    text = item.get('text', '')
                    explanation = item.get('explanation', '')
                    example = item.get('example', '')
                    difficulty = item.get('difficulty', 2)

                    # 将类别从复数形式转为单数
                    if category == 'grammar_points':
                        db_category = 'grammar_point'
                    elif category == 'phrases':
                        db_category = 'phrase'
                    elif category == 'collocations':
                        db_category = 'collocation'
                    else:
                        db_category = category

                    # 保存到数据库
                    self.user_model.add_knowledge_point(
                        knowledge_id,
                        knowledge_type,
                        db_category,
                        text,
                        explanation,
                        example,
                        difficulty
                    )

            return True
        except Exception as e:
            print(f"保存知识点到数据库时出错: {str(e)}")
            return False

    def decompose_sentence(self, sentence: str) -> List[str]:
        """将句子分解为不同层级，使用DeepSeek API实现"""
        prompt = f"""
        请将以下英语句子分解为由简到难的不同层级，以便用于逐层翻译学习:

        句子: {sentence}

        请提供3-4个层级，从最简单的主干逐步扩展到完整句子:
        1. 第一层: 最简单的主干结构(主谓或主谓宾)
        2. 第二层: 添加一些修饰或从句
        3. 第三层: 包含更多细节
        4. 第四层: 完整句子

        请以JSON格式返回，格式为:
        {{
            "layers": ["最简层级", "基础层级", "扩展层级", "完整句子"]
        }}

        注意层级之间应有明显的复杂度递增，但保持语法正确性。
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
                data = json.loads(content)
                return data.get("layers", [sentence])
            else:
                print(f"DeepSeek API 句子分解调用失败: {response.status_code}")
                # 如果API调用失败，返回原句作为唯一层级
                return [sentence]
        except Exception as e:
            print(f"DeepSeek API 句子分解异常: {str(e)}")
            return [sentence]

    def save_knowledge_point(self, knowledge_type, knowledge_id, text, explanation, example, difficulty, mode):
        """保存或更新知识点到数据库"""
        try:
            # 确定要操作的知识点类别
            if 'grammar' in knowledge_type:
                category = 'grammar_point'
            elif 'phrase' in knowledge_type:
                category = 'phrase'
            elif 'collocation' in knowledge_type:
                category = 'collocation'
            else:
                category = knowledge_type

            # 如果是添加新知识点，生成新ID
            if mode == 'add':
                import hashlib
                if not knowledge_id:
                    knowledge_id = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]

            # 保存到数据库
            result = self.user_model.add_knowledge_point(
                knowledge_id,
                knowledge_type,
                category,
                text,
                explanation,
                example,
                int(difficulty)
            )

            if result:
                return {
                    'id': knowledge_id,
                    'type': knowledge_type,
                    'category': category,
                    'text': text,
                    'explanation': explanation,
                    'example': example,
                    'difficulty': int(difficulty)
                }
            else:
                raise Exception("保存知识点失败")

        except Exception as e:
            print(f"保存知识点时出错: {str(e)}")
            raise

    def delete_knowledge_point(self, knowledge_type, knowledge_id):
        """从数据库删除知识点"""
        try:
            result = self.user_model.delete_knowledge_point(knowledge_id)
            return result
        except Exception as e:
            print(f"删除知识点时出错: {str(e)}")
            raise