import os
import uuid
import torch
import numpy as np
import pymysql
from PIL import Image
from datetime import datetime
from werkzeug.utils import secure_filename
from model import ImageTransformerModel
from torchvision.transforms import functional as T
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# 数据库配置
db_config = {
    'host': 'localhost',
    'user': 'root',  # 填入正确的数据库用户名
    'password': '',  # 填入正确的数据库密码
    'database': 'style_transfer',
    'charset': 'utf8mb4'
}

# 初始化登录管理器
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后再访问该页面'

# 配置上传文件相关
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 数据库连接
def get_db_connection():
    return pymysql.connect(**db_config)

# 表结构定义（需确保已手动创建表或使用 SQL 脚本）
def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password VARCHAR(200) NOT NULL,
                    name VARCHAR(60) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transfer_record (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    original_path VARCHAR(255) NOT NULL,
                    result_path VARCHAR(255) NOT NULL,
                    style VARCHAR(30) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user (id)
                )
            ''')
            conn.commit()

# 确保数据库初始化
init_db()

# 加载模型 - 保持不变
print("正在初始化模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
charcoal_model = ImageTransformerModel().train().to(device)
charcoal_weight = torch.load('models/style2.pt', map_location=torch.device('cpu'))
charcoal_model.load_state_dict(charcoal_weight)
watercolor_model = ImageTransformerModel().train().to(device)
watercolor_weight = torch.load('models/style3.pt', map_location=torch.device('cpu'))
watercolor_model.load_state_dict(watercolor_weight)
impression_model = ImageTransformerModel().train().to(device)
impression_weight = torch.load('models/style1.pt', map_location=torch.device('cpu'))
impression_model.load_state_dict(impression_weight)
print("模型加载完成！")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@login_manager.user_loader
def load_user(user_id):
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT * FROM user WHERE id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                user = User()
                user.id = user_data['id']
                user.email = user_data['email']
                user.name = user_data['name']
                return user
    return None


# 自定义用户类
class User(UserMixin):
    def __init__(self, id=None, email=None, name=None):
        self.id = id
        self.email = email
        self.name = name


# 路由
@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        with get_db_connection() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
                user_data = cursor.fetchone()

        if user_data and check_password_hash(user_data['password'], password):
            user = User(
                id=user_data['id'],
                email=user_data['email'],
                name=user_data['name']
            )
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('邮箱或密码错误', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')

        with get_db_connection() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
                user_exists = cursor.fetchone()

        if user_exists:
            flash('该邮箱已被注册', 'error')
            return render_template('register.html')

        if len(password) < 8:
            flash('密码长度至少为8个字符', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)

        with get_db_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO user (email, password, name) VALUES (%s, %s, %s)",
                        (email, hashed_password, name)
                    )
                conn.commit()

                # 获取新创建的用户ID
                user_id = cursor.lastrowid

                # 创建用户对象并登录
                user = User(
                    id=user_id,
                    email=email,
                    name=name
                )
                login_user(user)
                flash('注册成功！', 'success')
                return redirect(url_for('index'))
            except Exception as e:
                conn.rollback()
                flash(f'注册失败: {str(e)}', 'error')

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/history')
@login_required
def history():
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT 
                    COUNT(*) AS total_records,
                    SUM(CASE WHEN style = 'charcoal' THEN 1 ELSE 0 END) AS charcoal,
                    SUM(CASE WHEN style = 'watercolor' THEN 1 ELSE 0 END) AS watercolor,
                    SUM(CASE WHEN style = 'impression' THEN 1 ELSE 0 END) AS impression
                FROM transfer_record
                WHERE user_id = %s
            ''', (current_user.id,))
            stats_data = cursor.fetchone()

    # 创建一个包含 styles 属性的字典
    stats = {
        'total_records': stats_data['total_records'] if stats_data else 0,
        'styles': {
            'charcoal': stats_data['charcoal'] if stats_data else 0,
            'watercolor': stats_data['watercolor'] if stats_data else 0,
            'impression': stats_data['impression'] if stats_data else 0
        }
    }

    return render_template('history.html', stats=stats)


@app.route('/convert', methods=['POST'])
@login_required
def convert_image():
    if 'file' not in request.files:
        return {'error': '未上传文件'}, 400

    model = None
    file = request.files['file']
    style = request.form.get('style')

    if not style:
        return {'error': '未选择风格'}, 400

    if style == 'charcoal':
        model = charcoal_model
    elif style == 'watercolor':
        model = watercolor_model
    elif style == 'impression':
        model = impression_model
    else:
        return {'error': '无效的风格选择'}, 400

    if file.filename == '':
        return {'error': '未选择文件'}, 400

    if not allowed_file(file.filename):
        return {'error': '不支持的文件类型，请上传JPG或PNG格式的图片'}, 400

    try:
        # 保存原始文件
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()

        original_filename = f"original_{timestamp}_{random_id}.{file_ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # 加载图片内容 - 保持风格转换代码不变
        image_pil = Image.open(original_path).convert("RGB")
        image_np = np.array(image_pil)
        image_t = T.to_tensor(image_np)
        image_t.unsqueeze_(0)
        image_t = image_t.to(device)

        # 进行风格转换
        with torch.no_grad():
            transformed_t = model(image_t)
        transformed_t.squeeze_(0)
        transformed_t = transformed_t.detach().cpu()
        transformed_image = T.to_pil_image(transformed_t)

        # 保存结果
        result_filename = f"result_{timestamp}_{style}_{random_id}.{file_ext}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        transformed_image.save(result_path)

        # 添加到数据库
        original_url = url_for('static', filename=f'uploads/{original_filename}')
        result_url = url_for('static', filename=f'results/{result_filename}')

        with get_db_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO transfer_record (user_id, original_path, result_path, style) VALUES (%s, %s, %s, %s)",
                        (current_user.id, original_url, result_url, style)
                    )
                conn.commit()
                record_id = cursor.lastrowid
                return {
                    'success': True,
                    'result_url': result_url,
                    'original_url': original_url,
                    'style_used': style,
                    'record_id': record_id
                }
            except Exception as e:
                conn.rollback()
                print(f"数据库插入错误: {e}")
                raise

    except Exception as e:
        print(f"转换过程发生错误: {e}")
        return {'error': f'处理图片时发生错误: {str(e)}'}, 500

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/history')
@login_required
def get_history():
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT 
                    id,
                    original_path AS original,
                    result_path AS result,
                    style,
                    DATE(created_at) AS date
                FROM transfer_record
                WHERE user_id = %s
                ORDER BY created_at DESC
            ''', (current_user.id,))
            records = cursor.fetchall()

    return jsonify(records)


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
@login_required
def delete_history_item(record_id):
    with get_db_connection() as conn:
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                # 首先查询记录以获取文件路径
                cursor.execute("SELECT original_path, result_path FROM transfer_record WHERE id = %s AND user_id = %s", (record_id, current_user.id))
                record = cursor.fetchone()

                if not record:
                    return jsonify({"success": False, "error": "未找到记录或删除失败"}), 404

                # 尝试删除文件
                original_path = record['original_path'].lstrip('/')
                result_path = record['result_path'].lstrip('/')

                if os.path.exists(original_path):
                    os.remove(original_path)
                if os.path.exists(result_path):
                    os.remove(result_path)

                # 从数据库中删除
                cursor.execute("DELETE FROM transfer_record WHERE id = %s", (record_id,))
                conn.commit()

            return jsonify({"success": True})
        except Exception as e:
            conn.rollback()
            print(f"删除失败: {e}")
            return jsonify({"success": False, "error": "删除失败"}), 500


@app.route('/api/user')
@login_required
def get_user_info():
    return jsonify({
        "loggedIn": True,
        "email": current_user.email,
        "name": current_user.name,
        "id": current_user.id
    })


@app.route('/api/stats')
@login_required
def get_user_stats():
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 获取总记录数和各类风格数
            cursor.execute('''
                SELECT 
                    COUNT(*) AS total_records,
                    SUM(CASE WHEN style = 'charcoal' THEN 1 ELSE 0 END) AS charcoal,
                    SUM(CASE WHEN style = 'watercolor' THEN 1 ELSE 0 END) AS watercolor,
                    SUM(CASE WHEN style = 'impression' THEN 1 ELSE 0 END) AS impression
                FROM transfer_record
                WHERE user_id = %s
            ''', (current_user.id,))
            style_stats = cursor.fetchone()

            # 获取每日转换数据
            cursor.execute('''
                SELECT 
                    DATE(created_at) AS date,
                    COUNT(*) AS count
                FROM transfer_record
                WHERE user_id = %s
                GROUP BY DATE(created_at)
                ORDER BY date
            ''', (current_user.id,))
            daily_stats = cursor.fetchall()

    return jsonify({
        'total_records': style_stats['total_records'],
        'styles': {
            'charcoal': style_stats['charcoal'],
            'watercolor': style_stats['watercolor'],
            'impression': style_stats['impression']
        },
        'daily_data': daily_stats
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件大小不能超过5MB"}), 413


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)