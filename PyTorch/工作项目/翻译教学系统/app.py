from flask import Flask, render_template, request, jsonify, session
from services.teaching import TeachingService
from services.correction import CorrectionService
from database.models import UserModel
from models.knowledge import KnowledgeExtractor
from models.translator import TranslationModel
import os

app = Flask(__name__)
app.secret_key = 'your_very_complex_and_unique_secret_key_here'

# 确保data文件夹存在
os.makedirs('data', exist_ok=True)

# 初始化服务
teaching_service = TeachingService()
translate_service = TranslationModel()
correction_service = CorrectionService()
user_model = UserModel()
knowledge_extractor = KnowledgeExtractor()

# 创建数据库表
user_model.create_tables()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # 检查用户名是否已存在
    if user_model.get_user_by_username(username):
        return jsonify({'error': '用户名已存在', 'success': False}), 400

    # 注册新用户
    if user_model.add_user(username, password):
        return jsonify({'message': '注册成功！请登录', 'success': True}), 200
    else:
        return jsonify({'error': '注册失败，请稍后再试', 'success': False}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 验证用户
    user = user_model.authenticate_user(username, password)
    if not user:
        return jsonify({'error': '用户名或密码错误', 'success': False}), 401

    # 登录成功，将用户信息保存到会话
    session['user_id'] = user['id']
    session['username'] = user['username']
    return jsonify({'message': '登录成功', 'username': user['username'], 'success': True}), 200


@app.route('/get_random_sentence', methods=['POST'])
def get_random_sentence():
    translation_model = TranslationModel()
    sentence_data = translation_model.get_random_sentences()
    return jsonify(sentence_data)


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    chinese_text = data['text']
    english_text = translate_service.translate(chinese_text)
    mode = data['mode']

    if mode == 'prompt':
        result = translate_service.get_initial_letters(english_text)
        result = {
            'source': chinese_text,
            'reference': english_text,
            'prompt': result
        }
    elif mode == 'independent':
        hint_level = data.get('hint_level', 0)
        result = translate_service.get_initial_letters(english_text, hint_level)
        result = {
            'source': chinese_text,
            'reference': english_text,
            'prompt': result
        }
    elif mode == 'layered':
        layer_index = data.get('layer_index', 0)
        result = teaching_service.layered_translation(chinese_text, layer_index)
    else:  # full_sentence
        result = teaching_service.full_sentence_translation(chinese_text)

    return jsonify(result)


@app.route('/correct', methods=['POST'])
def correct():
    data = request.json
    user_translation = data['user_translation']
    target_translation = data['target_translation']
    source_text = data.get('source_text', '')
    translation_type = data.get('translation_type', 'unknown')

    # 调用批改服务
    result = correction_service.correct_translation(user_translation, target_translation)

    # 获取带颜色标记的批改结果，方便前端展示
    colored_result = correction_service.get_colored_correction(user_translation, target_translation)
    result['colored_correction'] = colored_result

    # 如果用户已登录，保存记录
    if 'user_id' in session:
        # 添加翻译记录
        user_model.add_translation_record(
            session['user_id'],
            source_text,
            target_translation,
            user_translation,
            result['accuracy'],
            translation_type
        )

    # 构建响应
    response = {
        'detailed_correction': result['detailed_correction'],
        'accuracy': result['accuracy'],
        'user_translation': result['user_translation'],
        'target_translation': result['target_translation'],
        'colored_correction': result['colored_correction'],
        'evaluation_summary': _generate_evaluation_summary(result)
    }

    # 添加额外分析，如果存在
    if 'deepseek_analysis' in result:
        response['deepseek_analysis'] = result['deepseek_analysis']

    return jsonify(response)


def _generate_evaluation_summary(result):
    """生成翻译评估总结"""
    accuracy = result['accuracy']

    # 构建评价级别
    if accuracy >= 90:
        level = "优秀"
    elif accuracy >= 75:
        level = "良好"
    elif accuracy >= 60:
        level = "及格"
    else:
        level = "需要改进"

    # 构建评价内容
    evaluation = f"总体评分：{level}（{accuracy}%）\n"

    # 添加其他评价内容，如果存在
    if 'deepseek_analysis' in result:
        analysis = result['deepseek_analysis']

        # 添加语法错误
        if 'grammar_error_details' in analysis and analysis['grammar_error_details']:
            evaluation += "\n语法错误详情：\n"
            for error in analysis['grammar_error_details'][:3]:  # 只显示前3个错误
                evaluation += f"- {error}\n"

        # 添加词汇分析
        if 'vocabulary_analysis' in analysis and analysis['vocabulary_analysis']:
            evaluation += f"\n词汇分析：\n{analysis['vocabulary_analysis']}\n"

    return evaluation


@app.route('/knowledge_points', methods=['GET', 'POST'])
def knowledge_points():
    data = request.json
    sentence = data['reference']

    # 调用知识点提取服务
    result = knowledge_extractor.extract_knowledge(sentence)

    # 获取句子分解层级
    sentence_layers = knowledge_extractor.decompose_sentence(sentence)

    # 构建返回结果
    formatted_result = {
        'status': 'success',
        'data': {
            'grammar_points': result.get('grammar_points', []),
            'phrases': result.get('phrases', []),
            'collocations': result.get('collocations', []),
            'sentence_layers': sentence_layers
        }
    }

    return jsonify(formatted_result)


@app.route('/user_progress', methods=['GET'])
def user_progress():
    if 'user_id' not in session:
        # 未登录时，返回默认值0
        return jsonify({
            'total_exercises': 0,
            'avg_accuracy': 0,
            'consecutive_days': 0,
            'achievements': []
        })

    progress = user_model.get_user_progress(session['user_id'])
    return jsonify(progress)


# 添加自定义句子的路由
@app.route('/add_custom_sentence', methods=['POST'])
def add_custom_sentence():
    data = request.json
    chinese = data.get('chinese')
    difficulty = data.get('difficulty', 2)  # 默认为中等难度

    if not chinese:
        return jsonify({'error': '请提供完整的中文句子'}), 400

    try:
        # 调用翻译模型的方法添加自定义句子
        translation_model = TranslationModel()
        english = translation_model.translate(chinese)
        translation_model.add_custom_sentence(chinese)

        # 生成首字母提示等信息
        sentence_data = {
            'source': chinese,
            'reference': english,
            'prompt': translation_model.get_initial_letters(english),
        }

        return jsonify({
            'success': True,
            'message': '句子添加成功',
            'sentence': sentence_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 保存知识点的路由
@app.route('/save_knowledge_point', methods=['POST'])
def save_knowledge_point():
    data = request.json
    knowledge_type = data.get('type')
    knowledge_id = data.get('id')
    text = data.get('text')
    explanation = data.get('explanation')
    example = data.get('example', '')
    difficulty = data.get('difficulty', 2)
    mode = data.get('mode', 'edit')  # 'edit' 或 'add'

    if not text or not explanation:
        return jsonify({'error': '请提供完整的知识点信息'}), 400

    try:
        # 调用知识点提取器的方法保存知识点
        result = knowledge_extractor.save_knowledge_point(
            knowledge_type,
            knowledge_id,
            text,
            explanation,
            example,
            difficulty,
            mode
        )

        return jsonify({
            'success': True,
            'message': '知识点保存成功',
            'knowledge': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 删除知识点的路由
@app.route('/delete_knowledge_point', methods=['POST'])
def delete_knowledge_point():
    data = request.json
    knowledge_type = data.get('type')
    knowledge_id = data.get('id')

    if not knowledge_type or knowledge_id is None:
        return jsonify({'error': '请提供完整的知识点信息'}), 400

    try:
        # 调用知识点提取器的方法删除知识点
        result = knowledge_extractor.delete_knowledge_point(knowledge_type, knowledge_id)

        return jsonify({
            'success': True,
            'message': '知识点删除成功'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)