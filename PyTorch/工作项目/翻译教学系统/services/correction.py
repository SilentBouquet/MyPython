import requests
import json
from typing import Dict, Any
from nltk.tokenize import word_tokenize
from config import CONFIG


class DeepSeekAPI:
    """DeepSeek API 调用接口"""

    def __init__(self):
        self.api_key = CONFIG['DEEPSEEK_API_KEY']
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def analyze_translation(self, user_translation: str, target_translation: str) -> Dict[str, Any]:
        """使用DeepSeek分析翻译结果"""
        prompt = f"""
        请分析并批改以下英语翻译，提供详细的语言分析：

        原文: {target_translation}
        翻译: {user_translation}

        请分析以下内容并以JSON格式返回:
        1. 翻译的详细批改，包括每个单词的正确性，替换建议等
        2. 语义相似度评分(0-1之间的小数)
        3. 词汇使用评价
        4. 语法和拼写错误
        5. 需要替换的词汇，以及推荐替换词

        每个单词的分析应包含：
        - 单词本身
        - 状态(correct/incorrect/extra/missing/synonym)
        - 修正建议(如有)
        - 错误类型(spelling/grammar/vocabulary/structure/omission/semantic)

        同时计算整体准确率，考虑：
        - 正确单词数量
        - 同义词替换(给予0.8权重)
        - 语义理解程度

        完整JSON格式应包含：
        {{
            "detailed_correction": [
                {{"word": "单词", "status": "状态", "correction": "修正", "error_type": "错误类型"}}
            ],
            "accuracy": 准确率百分比,
            "grammar_errors": 语法错误数量,
            "spelling_errors": 拼写错误数量,
            "semantic_score": 语义相似度(0-100),
            "vocabulary_analysis": "词汇分析",
            "synonym_pairs": [["原词1", "同义词1"], ["原词2", "同义词2"]],
            "grammar_error_details": ["错误1", "错误2"]
        }}
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
                return json.loads(content)
            else:
                print(f"DeepSeek API 调用失败: {response.status_code}, {response.text}")
                # 返回一个基本的结构，确保与原函数返回一致
                return self._generate_fallback_result(user_translation, target_translation)
        except Exception as e:
            print(f"DeepSeek API 调用异常: {str(e)}")
            return self._generate_fallback_result(user_translation, target_translation)

    def _generate_fallback_result(self, user_translation: str, target_translation: str) -> Dict[str, Any]:
        """生成备用结果，确保与原函数返回格式一致"""
        user_words = word_tokenize(user_translation.lower())

        # 创建一个简单的分析结果
        result = []
        for word in user_words:
            result.append({
                'word': word,
                'status': 'correct',  # 假设所有单词都是正确的
                'correction': None,
                'error_type': None
            })

        return {
            'detailed_correction': result,
            'accuracy': 50.0,  # 默认准确率
            'user_translation': user_translation,
            'target_translation': target_translation,
            'grammar_errors': 0,
            'spelling_errors': 0,
            'semantic_score': 50.0,
            'vocabulary_analysis': "无法进行分析",
            'synonym_pairs': [],
            'grammar_error_details': []
        }


class CorrectionService:
    def __init__(self):
        self.deepseek_api = DeepSeekAPI()

    def correct_translation(self, user_translation, target_translation):
        """批改翻译"""
        # 直接使用DeepSeek API进行翻译批改
        analysis_result = self.deepseek_api.analyze_translation(user_translation, target_translation)

        # 确保结果格式与原函数返回一致
        correction_result = {'detailed_correction': analysis_result.get('detailed_correction', []),
                             'accuracy': analysis_result.get('accuracy', 50.0), 'user_translation': user_translation,
                             'target_translation': target_translation,
                             'grammar_errors': analysis_result.get('grammar_errors', 0),
                             'spelling_errors': analysis_result.get('spelling_errors', 0),
                             'semantic_score': analysis_result.get('semantic_score', 50.0), 'deepseek_analysis': {
                'vocabulary_analysis': analysis_result.get('vocabulary_analysis', ''),
                'synonym_pairs': analysis_result.get('synonym_pairs', []),
                'grammar_error_details': analysis_result.get('grammar_error_details', [])
            }}

        # 添加DeepSeek特有的分析结果

        return correction_result

    def get_colored_correction(self, user_translation, target_translation):
        """获取带颜色标记的批改结果"""
        correction = self.correct_translation(user_translation, target_translation)

        colored_result = []
        for item in correction['detailed_correction']:
            if item['status'] == 'correct':
                colored_result.append(f"<span style='color:blue'>{item['word']}</span>")
            elif item['status'] == 'incorrect':
                error_info = f" ({item['correction']})"
                if item['error_type']:
                    error_info += f" [错误类型: {self._translate_error_type(item['error_type'])}]"
                colored_result.append(f"<span style='color:red'>{item['word']}{error_info}</span>")
            elif item['status'] == 'extra':
                colored_result.append(f"<span style='color:orange'>{item['word']} [多余]</span>")
            elif item['status'] == 'missing':
                colored_result.append(f"<span style='color:green'>{item['correction']} [缺失]</span>")
            elif item['status'] == 'synonym':
                colored_result.append(
                    f"<span style='color:purple'>{item['word']} [同义词: {item['correction']}]</span>")

        return ' '.join(colored_result)

    def _translate_error_type(self, error_type):
        """将错误类型翻译为中文"""
        error_types = {
            'spelling': '拼写错误',
            'grammar': '语法错误',
            'vocabulary': '词汇选择错误',
            'structure': '结构错误',
            'omission': '遗漏',
            'semantic': '语义错误'
        }
        return error_types.get(error_type, error_type)