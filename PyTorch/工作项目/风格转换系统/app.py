import os
import uuid
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from model import ImageTransformerModel
from torchvision.transforms import functional as T
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# 配置数据库
user = 'root'  # 引号里填写你的数据库用户名
password = 'yy040806'  # 引号里填写你的数据库密码
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://' + user + ':' + password + '@localhost''/style_transfer'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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


# 数据库模型定义
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship('TransferRecord', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'


class TransferRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_path = db.Column(db.String(255), nullable=False)
    result_path = db.Column(db.String(255), nullable=False)
    style = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TransferRecord {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'original': self.original_path,
            'result': self.result_path,
            'style': self.style,
            'date': self.created_at.strftime("%Y-%m-%d")
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
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

        # 验证邮箱是否已存在
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('该邮箱已被注册', 'error')
            return render_template('register.html')

        if len(password) < 8:
            flash('密码长度至少为8个字符', 'error')
            return render_template('register.html')

        # 创建新用户
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password, name=name)

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            flash('注册成功！', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
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
    # 获取统计数据
    stats = {
        'total_records': TransferRecord.query.filter_by(user_id=current_user.id).count(),
        'styles': {
            'charcoal': TransferRecord.query.filter_by(user_id=current_user.id, style='charcoal').count(),
            'watercolor': TransferRecord.query.filter_by(user_id=current_user.id, style='watercolor').count(),
            'impression': TransferRecord.query.filter_by(user_id=current_user.id, style='impression').count()
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

        record = TransferRecord(
            user_id=current_user.id,
            original_path=original_url,
            result_path=result_url,
            style=style
        )

        db.session.add(record)
        db.session.commit()

        return {
            'success': True,
            'result_url': result_url,
            'original_url': original_url,
            'style_used': style,
            'record_id': record.id
        }
    except Exception as e:
        db.session.rollback()
        print(f"转换过程发生错误: {e}")
        return {'error': f'处理图片时发生错误: {str(e)}'}, 500


@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/api/history')
@login_required
def get_history():
    records = TransferRecord.query.filter_by(user_id=current_user.id).order_by(TransferRecord.created_at.desc()).all()
    return jsonify([record.to_dict() for record in records])


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
@login_required
def delete_history_item(record_id):
    record = TransferRecord.query.filter_by(id=record_id, user_id=current_user.id).first()

    if not record:
        return jsonify({"success": False, "error": "未找到记录或删除失败"}), 404

    try:
        # 尝试删除对应的文件
        if os.path.exists(record.original_path.lstrip('/')):
            os.remove(record.original_path.lstrip('/'))
        if os.path.exists(record.result_path.lstrip('/')):
            os.remove(record.result_path.lstrip('/'))
    except Exception as e:
        print(f"删除文件失败: {e}")

    # 从数据库中删除
    db.session.delete(record)
    db.session.commit()

    return jsonify({"success": True})


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
    # 已有的统计数据
    total = TransferRecord.query.filter_by(user_id=current_user.id).count()
    charcoal = TransferRecord.query.filter_by(user_id=current_user.id, style='charcoal').count()
    watercolor = TransferRecord.query.filter_by(user_id=current_user.id, style='watercolor').count()
    impression = TransferRecord.query.filter_by(user_id=current_user.id, style='impression').count()

    # 计算每日转换数据
    from sqlalchemy import func
    daily_stats = db.session.query(
        func.date(TransferRecord.created_at).label('date'),
        func.count(TransferRecord.id).label('count')
    ).filter_by(user_id=current_user.id).group_by('date').order_by('date').all()

    daily_data = [{'date': str(item.date), 'count': item.count} for item in daily_stats]

    return jsonify({
        'total_records': total,
        'styles': {
            'charcoal': charcoal,
            'watercolor': watercolor,
            'impression': impression
        },
        'daily_data': daily_data
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件大小不能超过5MB"}), 413


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# 初始化数据库
def init_db():
    with app.app_context():
        db.create_all()
        # 创建测试用户(如果不存在)
        if not User.query.filter_by(email='user@example.com').first():
            test_user = User(
                email='user@example.com',
                password=generate_password_hash('password123'),
                name='Test User'
            )
            db.session.add(test_user)
            db.session.commit()


if __name__ == '__main__':
    init_db()  # 初始化数据库
    app.run(debug=True)