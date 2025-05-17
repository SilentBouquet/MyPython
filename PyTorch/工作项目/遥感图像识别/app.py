# app.py - 主应用文件
import datetime
import json
import os
import shutil
import time
import uuid
import cv2
import jwt
import numpy as np
import pandas as pd
import pymysql
import torch
from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MODEL_FOLDER'] = 'models/custom/'
app.config['RESULTS_FOLDER'] = 'static/results/'  # 添加结果图片存储目录
app.config['VIDEO_RESULTS_FOLDER'] = 'static/videos/'  # 添加视频结果存储目录
app.config['REALTIME_RESULTS_FOLDER'] = 'static/realtime/'  # 添加实时监测结果存储目录
app.config['JWT_SECRET'] = 'your_jwt_secret_key'
app.config['JWT_EXPIRATION'] = 3600 * 24 * 7  # 7天
app.config['TOKEN_EXPIRATION'] = 24 * 3600  # 24小时有效期
PASSWORD = ''  # 数据库密码

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)  # 确保结果目录存在
os.makedirs(app.config['VIDEO_RESULTS_FOLDER'], exist_ok=True)  # 确保视频结果目录存在
os.makedirs(app.config['REALTIME_RESULTS_FOLDER'], exist_ok=True)  # 确保实时监测结果目录存在

# 实时监测相关全局变量
realtime_sessions = {}  # 存储所有实时监测会话 {session_id: session_data}

# 实时监测会话结构
"""
{
    'user_id': 用户ID,
    'model_id': 模型ID,
    'model': YOLO模型实例,
    'camera': 摄像头实例,
    'start_time': 开始时间,
    'frame_count': 已处理帧数,
    'detection_count': 检测到的物体总数,
    'categories': 检测到的类别集合,
    'confidence_sum': 置信度总和,
    'status': 状态(running/stopped),
    'last_result': 最后一帧处理结果,
    'record_path': 录制视频路径(如果有)
}
"""


# 数据库连接
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password=PASSWORD,
        database='remote_sensing_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}


def allowed_file(filename):
    """
    检查文件是否为系统允许的类型
    """
    # 定义允许的文件扩展名
    ALLOWED_EXTENSIONS = {
        # 图像文件
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff',
        # 视频文件
        'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'
    }

    # 获取文件扩展名并转换为小写
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    # 日志记录
    app.logger.debug(
        f"文件类型检查 - 文件名: {filename}, 扩展名: {extension}, 是否允许: {extension in ALLOWED_EXTENSIONS}")

    return extension in ALLOWED_EXTENSIONS


# 用户注册
@app.route('/api/register', methods=['POST'])
def register():
    # 获取前端传来的JSON数据
    data = request.get_json()

    # 验证必须字段是否存在
    if not data or not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({
            'success': False,
            'message': '请提供用户名、邮箱和密码'
        }), 400

    username = data['username']
    email = data['email']
    password = data['password']
    terms_agreed = data.get('termsAgreed', False)

    # 验证是否同意服务条款
    if not terms_agreed:
        return jsonify({
            'success': False,
            'message': '请同意服务条款和隐私政策'
        }), 400

    # 连接到数据库
    conn = get_db_connection()
    cursor = conn.cursor()

    # 检查用户名是否已存在
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    if cursor.fetchone():
        conn.close()
        return jsonify({
            'success': False,
            'message': '用户名已被注册'
        }), 400

    # 检查邮箱是否已存在
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({
            'success': False,
            'message': '该邮箱已被注册'
        }), 400

    # 对密码进行加密
    hashed_password = generate_password_hash(password)

    try:
        # 修改: 插入语句与字段匹配数据库表结构
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """
            INSERT INTO users (username, email, password, organization, department, 
                             license_number, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (username, email, hashed_password, '默认组织', '默认部门',
             'DEFAULT-LICENSE-' + str(uuid.uuid4())[:8], current_time)
        )

        # 获取新用户ID
        user_id = cursor.lastrowid
        conn.commit()

        # 生成认证令牌
        token = generate_token(user_id, username)

        # 返回用户信息和令牌
        user_info = {
            'id': user_id,
            'username': username,
            'email': email,
            'organization': '默认组织',
            'department': '默认部门'
        }

        conn.close()
        return jsonify({
            'success': True,
            'message': '注册成功',
            'token': token,
            'user': user_info
        }), 201

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({
            'success': False,
            'message': f'注册失败: {str(e)}'
        }), 500


def generate_token(user_id, username):
    payload = {
        'exp': int((datetime.datetime.utcnow() + datetime.timedelta(seconds=app.config['JWT_EXPIRATION'])).timestamp()),
        'iat': int(datetime.datetime.utcnow().timestamp()),
        'sub': user_id,
        'username': username
    }
    token = jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')
    if isinstance(token, bytes):
        token = token.decode('utf-8')
    return token


# 用户登录
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')  # 可以是用户名或邮箱
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "请提供用户名和密码"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询用户
    cursor.execute("SELECT id, username, email, password, is_admin FROM users WHERE username = %s OR email = %s",
                   (username, username))
    user = cursor.fetchone()
    conn.close()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

    # 生成JWT令牌
    token = generate_token(user['id'], user['username'])

    return jsonify({
        "success": True,
        "message": "登录成功",
        "token": token,
        "user": {
            "id": user['id'],
            "username": user['username'],
            "email": user['email'],
            "is_admin": user['is_admin']
        }
    })


# 验证令牌中间件
def token_required(f):
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({"success": False, "message": "Token无效"}), 401

        if not token:
            return jsonify({"success": False, "message": "缺少Token"}), 401

        try:
            payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            user_id = payload['sub']

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, is_admin FROM users WHERE id = %s", (user_id,))
            current_user = cursor.fetchone()
            conn.close()

            if not current_user:
                return jsonify({"success": False, "message": "用户不存在"}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "message": "Token已过期"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"success": False, "message": "Token无效"}), 401

        return f(current_user, *args, **kwargs)

    decorated.__name__ = f.__name__
    return decorated


# 用户信息API
@app.route('/api/user/info', methods=['GET'])
@token_required
def get_user_info(current_user):
    # 从数据库获取用户的完整信息
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, email, name, phone, organization, department, bio, is_admin FROM users WHERE id = %s",
        (current_user['id'],)
    )
    user_data = cursor.fetchone()
    conn.close()

    if not user_data:
        return jsonify({"success": False, "message": "用户不存在"}), 404

    return jsonify({
        "success": True,
        "user": user_data
    })


# 图像上传接口
@app.route('/api/images/upload', methods=['POST'])
@token_required
def upload_image(current_user):
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "没有文件"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "未选择文件"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # 保存上传记录到数据库
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """INSERT INTO images 
                   (user_id, original_filename, storage_filename, upload_path, created_at) 
                   VALUES (%s, %s, %s, %s, NOW())""",
                (current_user['id'], filename, unique_filename, filepath)
            )
            conn.commit()
            image_id = cursor.lastrowid

            conn.close()

            return jsonify({
                "success": True,
                "message": "上传成功",
                "image": {
                    "id": image_id,
                    "filename": filename,
                    "path": filepath
                }
            })
        except Exception as e:
            conn.rollback()
            conn.close()
            return jsonify({"success": False, "message": f"保存记录失败: {str(e)}"}), 500

    return jsonify({"success": False, "message": "文件类型不允许"}), 400


# 图像处理接口
@app.route('/api/images/process', methods=['POST'])
@token_required
def process_image(current_user):
    data = request.json
    image_id = data.get('imageId')
    model_id = data.get('modelId')

    if not image_id or not model_id:
        return jsonify({"success": False, "message": "缺少必要参数"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取图像信息
    cursor.execute("SELECT * FROM images WHERE id = %s AND user_id = %s", (image_id, current_user['id']))
    image_data = cursor.fetchone()

    if not image_data:
        conn.close()
        return jsonify({"success": False, "message": "图像不存在或无权访问"}), 404

    # 获取模型信息
    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()

    if not model_data:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在"}), 404

    # 处理图像
    try:
        start_time = time.time()

        # 加载图像
        image_path = image_data['upload_path']
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图片：{image_path}")
            return

        # 将图片转换为 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载模型并处理
        model = YOLO(model_data['path'])

        # 假设model.process方法返回识别结果和准确度
        result = model.predict(img_rgb, conf=0.5)[0]

        boxes = result.boxes.xyxy  # 获取边界框坐标
        cls_ids = result.boxes.cls  # 获取类别 ID
        confidences = result.boxes.conf  # 获取置信度

        categories = []
        # 在图片上绘制边界框和标签
        for box, cls_id, conf in zip(boxes, cls_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            cls_name = model.names[int(cls_id)]
            categories.append(cls_name)
            label = f"{cls_name}: {conf:.2f}"
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存处理结果到静态目录
        result_filename = f"result_{uuid.uuid4().hex}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        result_path_db = result_path  # 存储在数据库中的路径
        cv2.imwrite(result_path, img)

        # 提取识别到的类别和目标数量
        object_count = len(categories)
        categories = list(set(categories))

        # 计算平均置信度
        if len(confidences) > 0:
            accuracy = torch.mean(confidences).item() * 100
        else:
            accuracy = 0.0

        # 计算处理时间
        processing_time = time.time() - start_time

        # 检查数据库中是否有object_count和categories字段
        try:
            # 记录处理历史
            result_file_size = os.path.getsize(result_path_db)
            cursor.execute(
                """INSERT INTO processing_history 
                   (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, object_count, categories, data_size) 
                   VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)""",
                (current_user['id'], image_id, model_id, result_path_db, accuracy, processing_time, object_count,
                 json.dumps(categories), result_file_size)
            )
        except Exception as db_error:
            # 如果数据库没有新字段，使用原来的插入语句
            cursor.execute(
                """INSERT INTO processing_history 
                   (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at) 
                   VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
                (current_user['id'], image_id, model_id, result_path_db, accuracy, processing_time)
            )

        conn.commit()
        history_id = cursor.lastrowid

        # 更新模型使用次数
        cursor.execute("UPDATE models SET usage_count = usage_count + 1 WHERE id = %s", (model_id,))
        conn.commit()

        conn.close()

        result_url = f"/static/results/{result_filename}"

        return jsonify({
            "success": True,
            "message": "处理成功",
            "result": {
                "id": history_id,
                "accuracy": accuracy,
                "processingTime": processing_time,
                "resultPath": result_path_db,
                "resultUrl": result_url,
                "categories": categories,
                "objectCount": object_count
            }
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"处理失败: {str(e)}"}), 500


# 批量图像处理接口
@app.route('/api/images/batch-process', methods=['POST'])
@token_required
def batch_process_images(current_user):
    data = request.json
    image_ids = data.get('imageIds')
    model_id = data.get('modelId')

    if not image_ids or not model_id or not isinstance(image_ids, list) or len(image_ids) == 0:
        return jsonify({"success": False, "message": "缺少必要参数或图像ID列表为空"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取模型信息
    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()

    if not model_data:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在"}), 404

    # 加载模型
    try:
        model = YOLO(model_data['path'])
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"加载模型失败: {str(e)}"}), 500

    # 处理结果列表
    results = []
    total_start_time = time.time()

    # 处理每张图片
    for image_id in image_ids:
        try:
            # 获取图像信息
            cursor.execute("SELECT * FROM images WHERE id = %s AND user_id = %s", (image_id, current_user['id']))
            image_data = cursor.fetchone()

            if not image_data:
                results.append({
                    "imageId": image_id,
                    "success": False,
                    "message": "图像不存在或无权访问"
                })
                continue

            # 处理图像
            start_time = time.time()

            # 加载图像
            image_path = image_data['upload_path']
            img = cv2.imread(image_path)
            if img is None:
                results.append({
                    "imageId": image_id,
                    "success": False,
                    "message": f"无法加载图片: {image_path}"
                })
                continue

            # 将图片转换为 RGB 格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 使用模型处理
            result = model.predict(img_rgb, conf=0.5)[0]

            boxes = result.boxes.xyxy  # 获取边界框坐标
            cls_ids = result.boxes.cls  # 获取类别 ID
            confidences = result.boxes.conf  # 获取置信度

            categories = []
            # 在图片上绘制边界框和标签
            for box, cls_id, conf in zip(boxes, cls_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[int(cls_id)]
                categories.append(cls_name)
                label = f"{cls_name}: {conf:.2f}"
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存处理结果到静态目录
            result_filename = f"result_{uuid.uuid4().hex}.png"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            result_path_db = result_path  # 存储在数据库中的路径
            cv2.imwrite(result_path, img)

            # 提取识别到的类别和目标数量
            object_count = len(categories)
            categories = list(set(categories))

            # 计算平均置信度
            if len(confidences) > 0:
                accuracy = torch.mean(confidences).item() * 100
            else:
                accuracy = 0.0

            # 计算处理时间
            processing_time = time.time() - start_time

            # 记录处理历史
            try:
                result_file_size = os.path.getsize(result_path_db)
                cursor.execute(
                    """INSERT INTO processing_history 
                       (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, object_count, categories, data_size) 
                       VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)""",
                    (current_user['id'], image_id, model_id, result_path_db, accuracy, processing_time, object_count,
                     json.dumps(categories), result_file_size)
                )
            except Exception as db_error:
                # 如果数据库没有新字段，使用原来的插入语句
                cursor.execute(
                    """INSERT INTO processing_history 
                       (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at) 
                       VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
                    (current_user['id'], image_id, model_id, result_path_db, accuracy, processing_time)
                )

            conn.commit()
            history_id = cursor.lastrowid

            result_url = f"/static/results/{result_filename}"

            # 添加结果到列表
            results.append({
                "imageId": image_id,
                "success": True,
                "result": {
                    "id": history_id,
                    "accuracy": accuracy,
                    "processingTime": processing_time,
                    "resultPath": result_path_db,
                    "resultUrl": result_url,
                    "categories": categories,
                    "objectCount": object_count
                }
            })

        except Exception as e:
            conn.rollback()
            results.append({
                "imageId": image_id,
                "success": False,
                "message": f"处理失败: {str(e)}"
            })

    # 更新模型使用次数 - 只增加一次，不管处理了多少图片
    cursor.execute("UPDATE models SET usage_count = usage_count + 1 WHERE id = %s", (model_id,))
    conn.commit()
    conn.close()

    # 计算总处理时间
    total_processing_time = time.time() - total_start_time

    return jsonify({
        "success": True,
        "message": f"批量处理完成，成功处理 {sum(1 for r in results if r['success'])} 张图片，失败 {sum(1 for r in results if not r['success'])} 张图片",
        "results": results,
        "totalProcessingTime": total_processing_time
    })


# 下载处理结果
@app.route('/api/images/download/<int:history_id>', methods=['GET'])
def download_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')

    if not token:
        return jsonify({"success": False, "message": "缺少Token"}), 401

    try:
        # 验证token
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user_id = payload['sub']

        # 检查用户是否存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return jsonify({"success": False, "message": "用户不存在"}), 401

        # 获取处理结果信息
        cursor.execute(
            """SELECT h.*, i.original_filename 
               FROM processing_history h 
               JOIN images i ON h.image_id = i.id 
               WHERE h.id = %s AND h.user_id = %s""",
            (history_id, user_id)
        )
        history = cursor.fetchone()
        conn.close()

        if not history:
            return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

        # 设置下载的文件名：原始文件名基础上加上_result后缀
        original_name = history['original_filename']
        name_parts = original_name.rsplit('.', 1)
        download_name = f"{name_parts[0]}_result.{name_parts[1]}" if len(name_parts) > 1 else f"{original_name}_result"

        return send_file(
            history['result_path'],
            as_attachment=True,
            download_name=download_name
        )
    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "message": "Token已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "message": "Token无效"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": f"下载失败: {str(e)}"}), 500


# 查看结果图片
@app.route('/api/images/view/<int:history_id>', methods=['GET'])
def view_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')

    if not token:
        return jsonify({"success": False, "message": "缺少Token"}), 401

    try:
        # 验证token
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user_id = payload['sub']

        # 检查用户是否存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return jsonify({"success": False, "message": "用户不存在"}), 401

        # 获取处理结果图片路径
        cursor.execute(
            """SELECT h.result_path 
               FROM processing_history h 
               WHERE h.id = %s AND h.user_id = %s""",
            (history_id, user_id)
        )
        history = cursor.fetchone()
        conn.close()

        if not history:
            return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

        # 检查文件是否存在
        if not history['result_path'] or not os.path.exists(history['result_path']):
            app.logger.error(f"结果文件不存在: {history['result_path']}")
            return jsonify({"success": False, "message": "结果文件不存在"}), 404

        # 直接返回图片文件，不作为附件（不下载）
        return send_file(
            history['result_path'],
            mimetype='image/png'
        )

    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "message": "Token已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "message": "Token无效"}), 401
    except Exception as e:
        app.logger.error(f"获取图片失败: {str(e)}")
        return jsonify({"success": False, "message": f"获取图片失败: {str(e)}"}), 500


# 获取所有模型
@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    # 获取分类和类型参数
    model_type = request.args.get('type', 'all')
    category = request.args.get('category', 'all')
    supports_video = request.args.get('supports_video', 'false').lower() == 'true'

    app.logger.info(
        f"获取模型列表 - 用户: {current_user['id']}, 类型: {model_type}, 分类: {category}, 是否支持视频: {supports_video}")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 构建查询
        query = """
        SELECT m.*, u.username as creator_name,
               (SELECT COUNT(*) FROM processing_history WHERE model_id = m.id) as usage_count,
               (SELECT COUNT(*) FROM favorite_models WHERE model_id = m.id AND user_id = %s) > 0 as is_favorite
        FROM models m
        LEFT JOIN users u ON m.creator_id = u.id
        WHERE 1=1
        """
        params = [current_user['id']]

        # 根据分类过滤
        if category == 'system':
            query += " AND m.is_system = 1"
        elif category == 'favorite':
            query += " AND m.id IN (SELECT model_id FROM favorite_models WHERE user_id = %s)"
            params.append(current_user['id'])
        elif category == 'custom':
            query += " AND m.creator_id = %s AND m.is_system = 0"
            params.append(current_user['id'])
        else:  # 'all' 或其他值
            query += " AND (m.is_system = 1 OR m.is_shared = 1 OR m.creator_id = %s)"
            params.append(current_user['id'])

        # 根据模型类型过滤
        if model_type != 'all':
            query += " AND m.type = %s"
            params.append(model_type)

        # 根据视频支持过滤
        if supports_video:
            query += " AND m.supports_video = 1"

        query += " ORDER BY m.created_at DESC"

        app.logger.debug(f"模型查询SQL: {query}")
        app.logger.debug(f"查询参数: {params}")

        cursor.execute(query, params)
        models = cursor.fetchall()

        app.logger.info(f"查询到 {len(models)} 个模型")

        result = []
        for model in models:
            # 检查模型是否支持视频
            supports_video = model['supports_video'] if model['supports_video'] is not None else True

            model_data = {
                'id': model['id'],
                'name': model['name'],
                'description': model['description'],
                'type': model['type'],
                'path': model['path'],
                'created_at': model['created_at'].strftime('%Y-%m-%d %H:%M:%S') if model['created_at'] else None,
                'creator_id': model['creator_id'],
                'creator_name': model['creator_name'],
                'is_system': bool(model['is_system']),
                'is_shared': bool(model['is_shared']),
                'example_image': model['example_image'],
                'result_example_image': model.get('result_example_image'),
                'usage_count': model['usage_count'],
                'accuracy': float(model['accuracy']) if model['accuracy'] is not None else None,
                'is_favorite': bool(model['is_favorite']),
                'supports_video': supports_video  # 添加视频支持标志
            }
            result.append(model_data)

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'models': result
        })
    except Exception as e:
        app.logger.error(f"获取模型列表出错: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取模型列表失败'
        }), 500


# 添加/上传新模型
@app.route('/api/models/add', methods=['POST'])
@token_required
def add_model(current_user):
    # 处理表单数据和文件上传
    name = request.form.get('name')
    model_type = request.form.get('type')
    description = request.form.get('description', '')
    is_shared = request.form.get('isShared') == 'true'
    publish_reason = request.form.get('publishReason', '')
    parameters = request.form.get('parameters', '')
    instructions = request.form.get('instructions', '')

    # 添加调试代码，输出所有接收到的表单键值
    print(f"接收到的表单数据: {request.form}")
    print(f"接收到的文件: {request.files}")
    print(f"模型类型(type)值: {model_type}")

    if not name or not model_type:
        return jsonify({"success": False, "message": f"缺少必要参数, name={name}, type={model_type}"}), 400

    if 'modelFile' not in request.files:
        return jsonify({"success": False, "message": "未提供模型文件"}), 400

    model_file = request.files['modelFile']
    if model_file.filename == '':
        return jsonify({"success": False, "message": "未选择模型文件"}), 400

    # 保存模型文件
    model_filename = secure_filename(model_file.filename)
    # 使用模型名称和时间戳生成唯一文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    name_for_file = secure_filename(name).lower().replace(' ', '_')
    unique_model_name = f"{name_for_file}_{timestamp}_{model_filename}"
    model_path = os.path.join(app.config['MODEL_FOLDER'], unique_model_name)
    model_file.save(model_path)

    # 处理初始示例图片（必填）
    example_image_path = None
    if 'exampleImage' in request.files:
        example_image = request.files['exampleImage']
        if example_image.filename != '':
            example_filename = secure_filename(example_image.filename)
            file_ext = example_filename.rsplit('.', 1)[1].lower() if '.' in example_filename else 'jpg'
            new_filename = f"{name_for_file}_{timestamp}_input.{file_ext}"
            example_image_dir = os.path.join('static', 'images')
            os.makedirs(example_image_dir, exist_ok=True)
            example_image_path = os.path.join(example_image_dir, new_filename)
            example_image.save(example_image_path)
        else:
            return jsonify({"success": False, "message": "请上传初始示例图片"}), 400
    else:
        return jsonify({"success": False, "message": "请上传初始示例图片"}), 400

    # 处理结果示例图片（必填）
    result_example_image_path = None
    if 'resultExampleImage' in request.files:
        result_example_image = request.files['resultExampleImage']
        if result_example_image.filename != '':
            result_example_filename = secure_filename(result_example_image.filename)
            file_ext = result_example_filename.rsplit('.', 1)[1].lower() if '.' in result_example_filename else 'jpg'
            new_filename = f"{name_for_file}_{timestamp}_result.{file_ext}"
            example_image_dir = os.path.join('static', 'images')
            os.makedirs(example_image_dir, exist_ok=True)
            result_example_image_path = os.path.join(example_image_dir, new_filename)
            result_example_image.save(result_example_image_path)
        else:
            return jsonify({"success": False, "message": "请上传结果示例图片"}), 400
    else:
        return jsonify({"success": False, "message": "请上传结果示例图片"}), 400

    # 保存模型信息到数据库
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 无论用户是否申请公开，都先保存为私有模型
        cursor.execute(
            """INSERT INTO models 
               (name, type, description, path, example_image, result_example_image, creator_id, is_shared, 
                is_system, accuracy, usage_count, parameters, instructions, created_at) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, 0, 0, 0, 0, %s, %s, NOW())""",
            (name, model_type, description, model_path, example_image_path, result_example_image_path,
             current_user['id'], parameters, instructions)
        )
        conn.commit()
        model_id = cursor.lastrowid

        # 如果用户申请公开，创建公开申请记录
        if is_shared and publish_reason:
            cursor.execute(
                """INSERT INTO model_publish_requests 
                   (model_id, user_id, reason, status, created_at) 
                   VALUES (%s, %s, %s, 'pending', NOW())""",
                (model_id, current_user['id'], publish_reason)
            )
            conn.commit()

            # 创建一条系统通知，提醒管理员有新的模型公开申请
            cursor.execute(
                """INSERT INTO notifications 
                   (user_id, title, content, type, is_read, created_at) 
                   SELECT id, '新的模型公开申请', %s, 'model_publish', 0, NOW() 
                   FROM users WHERE is_admin = 1""",
                (f"用户 {current_user['username']} 申请公开模型 \"{name}\"，请审核",)
            )
            conn.commit()

            # 创建一条通知给用户，告知申请已提交
            cursor.execute(
                """INSERT INTO notifications 
                   (user_id, title, content, type, is_read, created_at) 
                   VALUES (%s, '模型公开申请已提交', %s, 'model_publish', 0, NOW())""",
                (current_user['id'], f"您的模型 \"{name}\" 公开申请已提交，请等待管理员审核")
            )
            conn.commit()

        conn.close()

        return jsonify({
            "success": True,
            "message": "模型添加成功，公开申请已提交，请等待管理员审核" if is_shared else "模型添加成功",
            "model": {
                "id": model_id,
                "name": name,
                "type": model_type,
                "path": model_path,
                "isShared": False,  # 模型初始状态为私有
                "example_image": example_image_path,
                "result_example_image": result_example_image_path
            }
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"添加模型失败: {str(e)}"}), 500


# 添加/移除收藏模型
@app.route('/api/models/favorite/<int:model_id>', methods=['POST', 'DELETE'])
@token_required
def toggle_favorite_model(current_user, model_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查模型是否存在
        cursor.execute("SELECT id FROM models WHERE id = %s", (model_id,))
        model = cursor.fetchone()
        if not model:
            print(f"模型不存在: {model_id}")
            conn.close()
            return jsonify({"success": False, "message": "模型不存在"}), 404

        if request.method == 'POST':
            # 添加收藏前检查是否已收藏
            cursor.execute(
                "SELECT user_id, model_id FROM favorite_models WHERE user_id = %s AND model_id = %s",
                (current_user['id'], model_id)
            )
            if cursor.fetchone():
                conn.close()
                return jsonify({"success": True, "message": "该模型已在收藏列表中"})

            # 添加收藏
            cursor.execute(
                "INSERT INTO favorite_models (user_id, model_id, created_at) VALUES (%s, %s, NOW())",
                (current_user['id'], model_id)
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "已添加到收藏"})
        else:
            # 移除收藏
            cursor.execute(
                "DELETE FROM favorite_models WHERE user_id = %s AND model_id = %s",
                (current_user['id'], model_id)
            )

            # 即使没有删除任何行（可能之前就不是收藏的），也视为成功
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "已从收藏中移除"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"操作失败: {str(e)}"}), 500


# 编辑模型
@app.route('/api/models/<int:model_id>', methods=['PUT'])
@token_required
def update_model(current_user, model_id):
    data = request.json
    name = data.get('name')
    description = data.get('description')
    is_shared = data.get('isShared')

    if not name:
        return jsonify({"success": False, "message": "模型名称不能为空"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 检查模型是否存在且属于当前用户
    cursor.execute("SELECT id FROM models WHERE id = %s AND creator_id = %s", (model_id, current_user['id']))
    if not cursor.fetchone():
        conn.close()
        return jsonify({"success": False, "message": "模型不存在或无权编辑"}), 404

    try:
        cursor.execute(
            "UPDATE models SET name = %s, description = %s, is_shared = %s WHERE id = %s",
            (name, description, is_shared, model_id)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型更新成功"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"更新模型失败: {str(e)}"}), 500


# 获取历史记录
@app.route('/api/history', methods=['GET'])
@token_required
def get_history(current_user):
    # 分页参数
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('perPage', 10))
    offset = (page - 1) * per_page

    # 筛选参数
    time_filter = request.args.get('timeFilter', 'all')
    model_filter = request.args.get('modelFilter', 'all')
    type_filter = request.args.get('typeFilter', 'all')
    search_query = request.args.get('search', '')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建基础查询
    query = """
        SELECT h.*, i.original_filename, m.name as model_name, m.type as model_type
        FROM processing_history h
        JOIN images i ON h.image_id = i.id
        JOIN models m ON h.model_id = m.id
        WHERE h.user_id = %s
    """
    params = [current_user['id']]

    # 添加筛选条件
    if time_filter != 'all':
        if time_filter == 'today':
            query += " AND DATE(h.created_at) = CURDATE()"
        elif time_filter == 'week':
            query += " AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
        elif time_filter == 'month':
            query += " AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
        elif time_filter == 'custom':
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            if start_date and end_date:
                query += " AND DATE(h.created_at) BETWEEN %s AND %s"
                params.extend([start_date, end_date])

    if model_filter != 'all':
        query += " AND (m.id = %s OR m.type = %s)"
        params.extend([model_filter, model_filter])

    if type_filter != 'all':
        query += " AND m.type = %s"
        params.append(type_filter)

    if search_query:
        query += " AND (i.original_filename LIKE %s OR m.name LIKE %s)"
        search_param = f"%{search_query}%"
        params.extend([search_param, search_param])

    # 添加排序和分页
    query += " ORDER BY h.created_at DESC LIMIT %s OFFSET %s"
    params.extend([per_page, offset])

    # 执行查询
    cursor.execute(query, params)
    records = cursor.fetchall()

    # 获取总记录数（用于分页）
    count_query = query.split("ORDER BY")[0].replace(
        "SELECT h.*, i.original_filename, m.name as model_name, m.type as model_type", "SELECT COUNT(*)")
    count_params = params[:-2]  # 移除LIMIT和OFFSET参数
    cursor.execute(count_query, count_params)
    total_count = cursor.fetchone()['COUNT(*)']

    conn.close()

    # 处理结果
    for record in records:
        record['created_at'] = record['created_at'].isoformat()

    return jsonify({
        "success": True,
        "records": records,
        "pagination": {
            "total": total_count,
            "page": page,
            "perPage": per_page,
            "totalPages": (total_count + per_page - 1) // per_page
        }
    })


# 删除历史记录
@app.route('/api/history/<int:history_id>', methods=['DELETE'])
@token_required
def delete_history(current_user, history_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 检查记录是否存在且属于当前用户
    cursor.execute("SELECT result_path FROM processing_history WHERE id = %s AND user_id = %s",
                   (history_id, current_user['id']))
    record = cursor.fetchone()

    if not record:
        conn.close()
        return jsonify({"success": False, "message": "记录不存在或无权删除"}), 404

    try:
        # 删除结果文件
        if record['result_path'] and os.path.exists(record['result_path']):
            os.remove(record['result_path'])

        # 删除数据库记录
        cursor.execute("DELETE FROM processing_history WHERE id = %s", (history_id,))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "记录已删除"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


# 批量删除历史记录
@app.route('/api/history/batch/delete', methods=['POST'])
@token_required
def batch_delete_history(current_user):
    data = request.json
    history_ids = data.get('ids', [])

    if not history_ids:
        return jsonify({"success": False, "message": "未提供记录ID"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 获取要删除记录的结果文件路径
        format_strings = ','.join(['%s'] * len(history_ids))
        cursor.execute(
            f"SELECT id, result_path FROM processing_history WHERE id IN ({format_strings}) AND user_id = %s",
            (*history_ids, current_user['id']))
        records = cursor.fetchall()

        # 删除结果文件
        for record in records:
            if record['result_path'] and os.path.exists(record['result_path']):
                os.remove(record['result_path'])

        # 删除数据库记录
        valid_ids = [record['id'] for record in records]
        if valid_ids:
            format_strings = ','.join(['%s'] * len(valid_ids))
            cursor.execute(f"DELETE FROM processing_history WHERE id IN ({format_strings})", valid_ids)
            conn.commit()

        conn.close()

        return jsonify({
            "success": True,
            "message": f"成功删除{len(valid_ids)}条记录",
            "deletedCount": len(valid_ids)
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"批量删除失败: {str(e)}"}), 500


# 获取统计概览数据
@app.route('/api/statistics/overview', methods=['GET'])
@token_required
def get_statistics_overview(current_user):
    time_range = request.args.get('range', 'week')

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        if not start_date or not end_date:
            return jsonify({"success": False, "message": "自定义范围需要提供开始和结束日期"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建日期条件
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    elif time_range == 'month':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
    elif time_range == 'quarter':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
    elif time_range == 'year':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
    elif time_range == 'custom':
        date_condition = "AND DATE(created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    # 获取处理总量
    cursor.execute(f"SELECT COUNT(*) as total FROM processing_history WHERE user_id = %s {date_condition}", params)
    total_processed = cursor.fetchone()['total']

    # 获取平均处理时间
    cursor.execute(
        f"SELECT AVG(processing_time) as avg_time FROM processing_history WHERE user_id = %s {date_condition}", params)
    avg_time = cursor.fetchone()['avg_time'] or 0

    # 获取平均精度
    cursor.execute(f"SELECT AVG(accuracy) as avg_accuracy FROM processing_history WHERE user_id = %s {date_condition}",
                   params)
    avg_accuracy = cursor.fetchone()['avg_accuracy'] or 0

    # 获取数据量（这里假设有一个字段存储处理的数据大小）
    cursor.execute(f"SELECT SUM(data_size) as total_size FROM processing_history WHERE user_id = %s {date_condition}",
                   params)
    result = cursor.fetchone()
    total_size = result['total_size'] if result and result['total_size'] else 0

    # 获取时间序列数据（用于显示趋势图）
    time_data = []
    if time_range == 'week':
        cursor.execute(
            """SELECT DATE(created_at) as date, COUNT(*) as count, AVG(processing_time) as avg_time, 
               AVG(accuracy) as avg_accuracy, SUM(data_size) as data_size
               FROM processing_history 
               WHERE user_id = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
               GROUP BY DATE(created_at)
               ORDER BY date""",
            (current_user['id'],)
        )
        time_data = cursor.fetchall()

    conn.close()

    # 处理时间序列数据
    for item in time_data:
        item['date'] = item['date'].isoformat() if hasattr(item['date'], 'isoformat') else str(item['date'])
        item['avg_time'] = float(item['avg_time']) if item['avg_time'] else 0
        item['avg_accuracy'] = float(item['avg_accuracy']) if item['avg_accuracy'] else 0
        item['data_size'] = float(item['data_size']) if item['data_size'] else 0

    return jsonify({
        "success": True,
        "overview": {
            "totalProcessed": total_processed,
            "avgTime": round(avg_time, 2) if avg_time else 0,
            "avgAccuracy": round(avg_accuracy, 2) if avg_accuracy else 0,
            "totalSize": round(total_size / 1024 / 1024, 2) if total_size else 0,  # 转换为MB
            "timeData": time_data
        }
    })


# 获取模型使用统计
@app.route('/api/statistics/models', methods=['GET'])
@token_required
def get_model_statistics(current_user):
    time_range = request.args.get('range', 'week')
    chart_type = request.args.get('type', 'usage')  # usage, accuracy, time

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建日期条件
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    elif time_range == 'month':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
    elif time_range == 'quarter':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
    elif time_range == 'year':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
    elif time_range == 'custom':
        date_condition = "AND DATE(h.created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    # 构建查询
    if chart_type == 'usage':
        query = f"""
            SELECT m.id, m.name, COUNT(*) as value
            FROM processing_history h
            JOIN models m ON h.model_id = m.id
            WHERE h.user_id = %s {date_condition}
            GROUP BY m.id, m.name
            ORDER BY value DESC
            LIMIT 5
        """
    elif chart_type == 'accuracy':
        query = f"""
            SELECT m.id, m.name, AVG(h.accuracy) as value
            FROM processing_history h
            JOIN models m ON h.model_id = m.id
            WHERE h.user_id = %s {date_condition}
            GROUP BY m.id, m.name
            ORDER BY value DESC
            LIMIT 5
        """
    else:  # time
        query = f"""
            SELECT m.id, m.name, AVG(h.processing_time) as value
            FROM processing_history h
            JOIN models m ON h.model_id = m.id
            WHERE h.user_id = %s {date_condition}
            GROUP BY m.id, m.name
            ORDER BY value ASC
            LIMIT 5
        """

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    # 处理结果
    labels = []
    data = []
    for result in results:
        labels.append(result['name'])
        data.append(float(result['value']) if result['value'] else 0)

    return jsonify({
        "success": True,
        "chartData": {
            "labels": labels,
            "data": data,
            "type": chart_type
        }
    })


# 获取类别分布统计
@app.route('/api/statistics/categories', methods=['GET'])
@token_required
def get_category_statistics(current_user):
    time_range = request.args.get('range', 'week')

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建日期条件
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    elif time_range == 'month':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
    elif time_range == 'quarter':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
    elif time_range == 'year':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
    elif time_range == 'custom':
        date_condition = "AND DATE(h.created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    # 查询类别分布
    query = f"""
        SELECT m.type as category, COUNT(*) as count
        FROM processing_history h
        JOIN models m ON h.model_id = m.id
        WHERE h.user_id = %s {date_condition}
        GROUP BY m.type
        ORDER BY count DESC
    """

    cursor.execute(query, params)
    categories = cursor.fetchall()

    conn.close()

    # 计算百分比
    total = sum(item['count'] for item in categories)

    category_data = []
    for item in categories:
        percentage = (item['count'] / total * 100) if total > 0 else 0
        category_data.append({
            "category": item['category'],
            "percentage": round(percentage, 1)
        })

    # 转换成前端需要的格式
    labels = [item["category"] for item in category_data]
    data = [item["percentage"] for item in category_data]

    return jsonify({
        "success": True,
        "chartData": {
            "labels": labels,
            "data": data
        }
    })


# 获取时间趋势数据
@app.route('/api/statistics/trend', methods=['GET'])
@token_required
def get_trend_statistics(current_user):
    time_range = request.args.get('range', 'week')
    metric = request.args.get('metric', 'count')  # count, accuracy, time
    interval = request.args.get('interval', 'day')  # day, week, month

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 根据不同的时间范围和间隔设置分组和条件
    date_format = "%Y-%m-%d"  # 默认按天
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
        if interval == 'week':
            date_format = "%x-%v"  # ISO标准: 年-周
    elif time_range == 'month':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
        if interval == 'week':
            date_format = "%x-%v"
        elif interval == 'month':
            date_format = "%Y-%m"
    elif time_range == 'quarter':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
        if interval == 'week':
            date_format = "%x-%v"
        elif interval == 'month':
            date_format = "%Y-%m"
    elif time_range == 'year':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
        if interval == 'month':
            date_format = "%Y-%m"
    elif time_range == 'custom':
        date_condition = "AND DATE(created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

        # 根据自定义日期范围的长度选择合适的分组
        if start_date and end_date:
            start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            days_diff = (end - start).days

            if days_diff > 60:
                interval = 'month'
                date_format = "%Y-%m"
            elif days_diff > 14:
                interval = 'week'
                date_format = "%x-%v"

    # 构建查询
    select_clause = ""
    if metric == 'count':
        select_clause = "COUNT(*) as value"
    elif metric == 'accuracy':
        select_clause = "AVG(accuracy) as value"
    elif metric == 'time':
        select_clause = "AVG(processing_time) as value"

    query = f"""
        SELECT DATE_FORMAT(created_at, '{date_format}') as period, {select_clause}
        FROM processing_history
        WHERE user_id = %s {date_condition}
        GROUP BY period
        ORDER BY MIN(created_at)
    """

    cursor.execute(query, params)
    trend_data = cursor.fetchall()

    conn.close()

    # 处理结果
    labels = []
    data = []
    for item in trend_data:
        labels.append(item['period'])
        data.append(float(item['value']) if item['value'] else 0)

    return jsonify({
        "success": True,
        "chartData": {
            "labels": labels,
            "data": data,
            "metric": metric,
            "interval": interval
        }
    })


# 获取地理分布数据
@app.route('/api/statistics/geo', methods=['GET'])
@token_required
def get_geo_statistics(current_user):
    time_range = request.args.get('range', 'week')
    data_type = request.args.get('type', 'count')  # count, accuracy, time

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建日期条件
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    elif time_range == 'month':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
    elif time_range == 'quarter':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
    elif time_range == 'year':
        date_condition = "AND h.created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
    elif time_range == 'custom':
        date_condition = "AND DATE(h.created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    # 构建查询
    select_clause = ""
    if data_type == 'count':
        select_clause = "COUNT(*) as value"
    elif data_type == 'accuracy':
        select_clause = "AVG(h.accuracy) as value"
    elif data_type == 'time':
        select_clause = "AVG(h.processing_time) as value"

    # 注意：这里假设图像表中有location_lat和location_lng字段记录地理位置
    query = f"""
        SELECT i.location_lat as lat, i.location_lng as lng, {select_clause}
        FROM processing_history h
        JOIN images i ON h.image_id = i.id
        WHERE h.user_id = %s {date_condition}
        AND i.location_lat IS NOT NULL AND i.location_lng IS NOT NULL
        GROUP BY i.location_lat, i.location_lng
    """

    cursor.execute(query, params)
    geo_data = cursor.fetchall()

    conn.close()

    # 处理结果，转换为GeoJSON格式
    features = []
    for item in geo_data:
        features.append({
            "type": "Feature",
            "properties": {
                "value": float(item['value']) if item['value'] else 0
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(item['lng']), float(item['lat'])]
            }
        })

    return jsonify({
        "success": True,
        "geoData": {
            "type": "FeatureCollection",
            "features": features
        }
    })


# 获取多维对比分析数据
@app.route('/api/statistics/comparison', methods=['GET'])
@token_required
def get_comparison_statistics(current_user):
    model1_id = request.args.get('model1')
    model2_id = request.args.get('model2')

    if not model1_id or not model2_id:
        return jsonify({"success": False, "message": "需要提供两个模型ID进行比较"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取模型信息
    cursor.execute("SELECT id, name FROM models WHERE id IN (%s, %s)", (model1_id, model2_id))
    models = cursor.fetchall()

    if len(models) < 2:
        conn.close()
        return jsonify({"success": False, "message": "一个或两个模型不存在"}), 404

    # 维度定义
    dimensions = [
        "处理速度", "精确度", "数据量", "多模型支持", "类别识别", "批处理效率"
    ]

    # 构建对比数据（这里使用模拟数据，实际应从处理历史中计算）
    model1_data = []
    model2_data = []

    # 处理速度 - 计算平均处理时间的倒数（时间越短越好）
    cursor.execute(
        """SELECT 
           (SELECT 100 - AVG(processing_time) * 10 FROM processing_history WHERE model_id = %s AND user_id = %s) as model1,
           (SELECT 100 - AVG(processing_time) * 10 FROM processing_history WHERE model_id = %s AND user_id = %s) as model2
        """,
        (model1_id, current_user['id'], model2_id, current_user['id'])
    )
    speed_data = cursor.fetchone()
    model1_data.append(float(speed_data['model1']) if speed_data['model1'] else 75)
    model2_data.append(float(speed_data['model2']) if speed_data['model2'] else 85)

    # 精确度 - 直接使用平均精度
    cursor.execute(
        """SELECT 
           (SELECT AVG(accuracy) FROM processing_history WHERE model_id = %s AND user_id = %s) as model1,
           (SELECT AVG(accuracy) FROM processing_history WHERE model_id = %s AND user_id = %s) as model2
        """,
        (model1_id, current_user['id'], model2_id, current_user['id'])
    )
    accuracy_data = cursor.fetchone()
    model1_data.append(float(accuracy_data['model1']) if accuracy_data['model1'] else 92)
    model2_data.append(float(accuracy_data['model2']) if accuracy_data['model2'] else 95)

    # 对于其他维度，使用模拟数据
    # 数据量
    model1_data.append(85)
    model2_data.append(75)

    # 多模型支持
    model1_data.append(70)
    model2_data.append(95)

    # 类别识别
    model1_data.append(90)
    model2_data.append(85)

    # 批处理效率
    model1_data.append(80)
    model2_data.append(90)

    conn.close()

    return jsonify({
        "success": True,
        "comparisonData": {
            "dimensions": dimensions,
            "models": [models[0]['name'], models[1]['name']],
            "data": [model1_data, model2_data]
        }
    })


# 获取智能洞察
@app.route('/api/statistics/insights', methods=['GET'])
@token_required
def get_insights(current_user):
    time_range = request.args.get('range', 'week')

    # 自定义日期范围
    start_date = None
    end_date = None
    if time_range == 'custom':
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        if not start_date or not end_date:
            return jsonify({"success": False, "message": "自定义范围需要提供开始和结束日期"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 构建日期条件
    date_condition = ""
    params = [current_user['id']]

    if time_range == 'week':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    elif time_range == 'month':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
    elif time_range == 'quarter':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
    elif time_range == 'year':
        date_condition = "AND created_at >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)"
    elif time_range == 'custom':
        date_condition = "AND DATE(created_at) BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    # 生成洞察（实际应用中应该基于数据分析生成）
    # 这里提供一些示例洞察
    insights = [
        {
            "title": "模型使用趋势",
            "content": "最近一周内，自定义模型的使用量增长了28%，其中自定义模型#12的精确度提高了2.3%，成为使用最多的自定义模型。",
            "level": "info"
        },
        {
            "title": "精确度异常",
            "content": "10月18日16:30-18:00期间，针对建筑区域类别的识别精确度下降了5.2%，可能与该时段处理的图像质量较低有关。",
            "level": "warning"
        },
        {
            "title": "识别类别分布变化",
            "content": "与上月相比，水体类别的识别比例上升了4.3%，植被类别下降了2.1%，表明近期处理的遥感图像区域类型发生了变化。",
            "level": "info"
        },
        {
            "title": "处理效率提升机会",
            "content": "批量处理超过15张图像时，平均每张处理时间可缩短0.4秒，建议用户尽可能采用批量上传方式提高效率。",
            "level": "success"
        },
        {
            "title": "地域分布特点",
            "content": "华东地区用户上传的图像精确度平均高于其他地区2.1%，可能与该地区遥感图像质量和分辨率较高有关。",
            "level": "info"
        }
    ]

    # 随机选择3-4个洞察（模拟刷新功能）
    import random
    selected_insights = random.sample(insights, min(len(insights), random.randint(3, 4)))

    # 添加智能推荐
    recommendation = {
        "title": "智能推荐",
        "content": "基于您的使用模式，建议尝试自定义模型#12处理植被区域的图像，预计可提高识别精确度约3.5%。",
        "level": "recommendation"
    }

    conn.close()

    return jsonify({
        "success": True,
        "insights": selected_insights,
        "recommendation": recommendation
    })


# 用户资料更新
@app.route('/api/user/update', methods=['PUT'])
@token_required
def update_user_profile(current_user):
    data = request.json

    # 验证必要字段
    if not data:
        return jsonify({"success": False, "message": "未提供更新数据"}), 400

    # 获取可更新的字段
    username = data.get('username')
    email = data.get('email')
    organization = data.get('organization')
    department = data.get('department')
    name = data.get('name')
    phone = data.get('phone')
    bio = data.get('bio')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查用户名和邮箱是否已被其他用户使用
        if username and username != current_user['username']:
            cursor.execute("SELECT id FROM users WHERE username = %s AND id != %s", (username, current_user['id']))
            if cursor.fetchone():
                conn.close()
                return jsonify({"success": False, "message": "用户名已被使用"}), 400

        if email and email != current_user['email']:
            cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, current_user['id']))
            if cursor.fetchone():
                conn.close()
                return jsonify({"success": False, "message": "邮箱已被使用"}), 400

        # 构建更新SQL语句
        update_fields = []
        params = []

        if username:
            update_fields.append("username = %s")
            params.append(username)
        if email:
            update_fields.append("email = %s")
            params.append(email)
        if organization:
            update_fields.append("organization = %s")
            params.append(organization)
        if department:
            update_fields.append("department = %s")
            params.append(department)
        if name:
            update_fields.append("name = %s")
            params.append(name)
        if phone:
            update_fields.append("phone = %s")
            params.append(phone)
        if bio:
            update_fields.append("bio = %s")
            params.append(bio)

        # 如果没有更新字段，直接返回成功
        if not update_fields:
            conn.close()
            return jsonify({"success": True, "message": "无更新内容"})

        # 添加更新时间和用户ID
        update_fields.append("updated_at = NOW()")
        params.append(current_user['id'])

        # 执行更新
        sql = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
        cursor.execute(sql, params)
        conn.commit()

        # 获取更新后的用户信息
        cursor.execute(
            "SELECT id, username, email, organization, department, name, phone, bio FROM users WHERE id = %s",
            (current_user['id'],))
        updated_user = cursor.fetchone()
        conn.close()

        return jsonify({
            "success": True,
            "message": "资料更新成功",
            "user": updated_user
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"更新失败: {str(e)}"}), 500


# 密码修改
@app.route('/api/user/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    data = request.json

    # 验证必要字段
    if not all(k in data for k in ['currentPassword', 'newPassword', 'confirmPassword']):
        return jsonify({"success": False, "message": "请提供当前密码和新密码"}), 400

    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    confirm_password = data.get('confirmPassword')

    # 验证新密码是否一致
    if new_password != confirm_password:
        return jsonify({"success": False, "message": "新密码与确认密码不一致"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证当前密码
    cursor.execute("SELECT password FROM users WHERE id = %s", (current_user['id'],))
    user = cursor.fetchone()

    if not user or not check_password_hash(user['password'], current_password):
        conn.close()
        return jsonify({"success": False, "message": "当前密码不正确"}), 400

    try:
        # 更新密码
        hashed_password = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = %s, updated_at = NOW() WHERE id = %s",
                       (hashed_password, current_user['id']))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "密码更新成功"
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"密码更新失败: {str(e)}"}), 500


# 获取用户通知
@app.route('/api/notifications', methods=['GET'])
@token_required
def get_notifications(current_user):
    # 获取分页参数
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('perPage', 10))
    offset = (page - 1) * per_page

    # 获取筛选参数
    status = request.args.get('status', 'all')  # all, read, unread
    count_only = request.args.get('count_only', 'false').lower() == 'true'  # 是否只返回计数

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取未读消息数量
    cursor.execute(
        "SELECT COUNT(*) as unread FROM notifications WHERE user_id = %s AND is_read = 0",
        (current_user['id'],)
    )
    unread_count = cursor.fetchone()['unread']

    # 如果只需要返回计数，则不查询完整的通知列表
    if count_only:
        conn.close()
        return jsonify({
            "success": True,
            "unreadCount": unread_count
        })

    # 构建查询条件
    where_clause = "WHERE user_id = %s"
    params = [current_user['id']]

    if status == 'read':
        where_clause += " AND is_read = 1"
    elif status == 'unread':
        where_clause += " AND is_read = 0"

    # 获取通知列表
    query = f"""
        SELECT id, title, content, type, is_read, created_at
        FROM notifications
        {where_clause}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    cursor.execute(query, params + [per_page, offset])
    notifications = cursor.fetchall()

    # 获取总数
    count_query = f"SELECT COUNT(*) as total FROM notifications {where_clause}"
    cursor.execute(count_query, params)
    total = cursor.fetchone()['total']

    conn.close()

    # 处理日期格式
    for notification in notifications:
        notification['created_at'] = notification['created_at'].isoformat() if hasattr(notification['created_at'],
                                                                                       'isoformat') else str(
            notification['created_at'])

    return jsonify({
        "success": True,
        "notifications": notifications,
        "unreadCount": unread_count,
        "pagination": {
            "total": total,
            "page": page,
            "perPage": per_page,
            "totalPages": (total + per_page - 1) // per_page
        }
    })


# 标记通知为已读
@app.route('/api/notifications/read/<int:notification_id>', methods=['POST'])
@token_required
def mark_notification_read(current_user, notification_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证通知是否属于当前用户
    cursor.execute(
        "SELECT id FROM notifications WHERE id = %s AND user_id = %s",
        (notification_id, current_user['id'])
    )
    notification = cursor.fetchone()

    if not notification:
        conn.close()
        return jsonify({"success": False, "message": "通知不存在或无权操作"}), 404

    try:
        # 标记为已读
        cursor.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = %s",
            (notification_id,)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "通知已标记为已读"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"操作失败: {str(e)}"}), 500


# 标记所有通知为已读
@app.route('/api/notifications/read-all', methods=['POST'])
@token_required
def mark_all_notifications_read(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 标记该用户所有通知为已读
        cursor.execute(
            "UPDATE notifications SET is_read = 1 WHERE user_id = %s AND is_read = 0",
            (current_user['id'],)
        )
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"已将{affected_rows}条通知标记为已读"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"操作失败: {str(e)}"}), 500


# 删除通知
@app.route('/api/notifications/<int:notification_id>', methods=['DELETE'])
@token_required
def delete_notification(current_user, notification_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证通知是否属于当前用户
    cursor.execute(
        "SELECT id FROM notifications WHERE id = %s AND user_id = %s",
        (notification_id, current_user['id'])
    )
    notification = cursor.fetchone()

    if not notification:
        conn.close()
        return jsonify({"success": False, "message": "通知不存在或无权操作"}), 404

    try:
        # 删除通知
        cursor.execute("DELETE FROM notifications WHERE id = %s", (notification_id,))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "通知已删除"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


# 删除已读通知
@app.route('/api/notifications/delete-read', methods=['DELETE'])
@token_required
def delete_read_notifications(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 删除该用户的所有已读通知
        cursor.execute(
            "DELETE FROM notifications WHERE user_id = %s AND is_read = 1",
            (current_user['id'],)
        )
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"已删除{affected_rows}条已读通知"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


# 获取权限申请列表
@app.route('/api/permissions', methods=['GET'])
@token_required
def get_permissions(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取用户的权限申请记录
    cursor.execute(
        """SELECT id, request_type, request_reason, status, created_at, updated_at
           FROM permission_requests
           WHERE user_id = %s
           ORDER BY created_at DESC""",
        (current_user['id'],)
    )
    permissions = cursor.fetchall()

    conn.close()

    # 处理日期格式
    for permission in permissions:
        permission['created_at'] = permission['created_at'].isoformat() if hasattr(permission['created_at'],
                                                                                   'isoformat') else str(
            permission['created_at'])
        permission['updated_at'] = permission['updated_at'].isoformat() if hasattr(permission['updated_at'],
                                                                                   'isoformat') else str(
            permission['updated_at'])

    return jsonify({
        "success": True,
        "permissions": permissions
    })


# 创建权限申请
@app.route('/api/permissions', methods=['POST'])
@token_required
def create_permission_request(current_user):
    data = request.json

    # 验证必要字段
    if not data or not all(k in data for k in ['requestType', 'requestReason']):
        return jsonify({"success": False, "message": "请提供申请类型和原因"}), 400

    request_type = data.get('requestType')
    request_reason = data.get('requestReason')
    storage_size = data.get('storageSize')  # 可选，仅当requestType为'storage'时需要

    # 验证申请类型
    valid_types = ['model', 'storage', 'api', 'other']
    if request_type not in valid_types:
        return jsonify({"success": False, "message": "无效的申请类型"}), 400

    # 对于存储空间申请，验证请求的存储大小
    if request_type == 'storage' and (not storage_size or storage_size < 10):
        return jsonify({"success": False, "message": "存储空间申请需要指定合理的存储大小"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查是否有相同类型的待处理申请
        cursor.execute(
            """SELECT id FROM permission_requests 
               WHERE user_id = %s AND request_type = %s AND status = 'pending'""",
            (current_user['id'], request_type)
        )

        if cursor.fetchone():
            conn.close()
            return jsonify({"success": False, "message": "已有相同类型的待处理申请"}), 400

        # 创建新申请
        additional_info = {}
        if request_type == 'storage':
            additional_info['storage_size'] = storage_size

        cursor.execute(
            """INSERT INTO permission_requests 
               (user_id, request_type, request_reason, additional_info, status, created_at) 
               VALUES (%s, %s, %s, %s, 'pending', NOW())""",
            (current_user['id'], request_type, request_reason, json.dumps(additional_info) if additional_info else None)
        )

        conn.commit()
        request_id = cursor.lastrowid

        # 创建一条通知
        cursor.execute(
            """INSERT INTO notifications 
               (user_id, title, content, type, is_read, created_at) 
               VALUES (%s, %s, %s, %s, 0, NOW())""",
            (current_user['id'], "权限申请已提交",
             f"您的{get_request_type_name(request_type)}申请已提交，等待管理员审核。", "info")
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "申请已提交",
            "requestId": request_id
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"申请提交失败: {str(e)}"}), 500


# 获取申请详情
@app.route('/api/permissions/<int:request_id>', methods=['GET'])
@token_required
def get_permission_detail(current_user, request_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证申请是否属于当前用户
    cursor.execute(
        """SELECT * FROM permission_requests 
           WHERE id = %s AND user_id = %s""",
        (request_id, current_user['id'])
    )

    permission = cursor.fetchone()
    conn.close()

    if not permission:
        return jsonify({"success": False, "message": "申请不存在或无权查看"}), 404

    # 处理日期格式
    permission['created_at'] = permission['created_at'].isoformat() if hasattr(permission['created_at'],
                                                                               'isoformat') else str(
        permission['created_at'])
    permission['updated_at'] = permission['updated_at'].isoformat() if hasattr(permission['updated_at'],
                                                                               'isoformat') else str(
        permission['updated_at'])

    # 解析额外信息
    if permission['additional_info']:
        try:
            permission['additional_info'] = json.loads(permission['additional_info'])
        except:
            pass

    return jsonify({
        "success": True,
        "permission": permission
    })


# 更新申请
@app.route('/api/permissions/<int:request_id>', methods=['PUT'])
@token_required
def update_permission_request(current_user, request_id):
    data = request.json

    if not data:
        return jsonify({"success": False, "message": "未提供更新数据"}), 400

    request_reason = data.get('requestReason')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证申请是否属于当前用户且状态为待处理
    cursor.execute(
        """SELECT id FROM permission_requests 
           WHERE id = %s AND user_id = %s AND status = 'pending'""",
        (request_id, current_user['id'])
    )

    permission = cursor.fetchone()

    if not permission:
        conn.close()
        return jsonify({"success": False, "message": "申请不存在、无权更新或已处理"}), 404

    try:
        # 更新申请原因
        cursor.execute(
            """UPDATE permission_requests 
               SET request_reason = %s, updated_at = NOW() 
               WHERE id = %s""",
            (request_reason, request_id)
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "申请已更新"
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"更新失败: {str(e)}"}), 500


# 取消申请
@app.route('/api/permissions/<int:request_id>', methods=['DELETE'])
@token_required
def cancel_permission_request(current_user, request_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证申请是否属于当前用户且状态为待处理
    cursor.execute(
        """SELECT id FROM permission_requests 
           WHERE id = %s AND user_id = %s AND status = 'pending'""",
        (request_id, current_user['id'])
    )

    permission = cursor.fetchone()

    if not permission:
        conn.close()
        return jsonify({"success": False, "message": "申请不存在、无权取消或已处理"}), 404

    try:
        # 删除申请
        cursor.execute("DELETE FROM permission_requests WHERE id = %s", (request_id,))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "申请已取消"
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"取消失败: {str(e)}"}), 500


# 辅助函数：获取请求类型的中文名称
def get_request_type_name(request_type):
    type_names = {
        'model': '高级模型使用权限',
        'storage': '增加存储空间',
        'api': 'API接口调用权限',
        'other': '其他权限'
    }
    return type_names.get(request_type, '未知类型')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login.html')
def login_html():
    return render_template('login.html')


@app.route('/register.html')
def register_html():
    return render_template('register.html')


@app.route('/dashboard.html')
def dashboard_html():
    return render_template('dashboard.html')


# 获取单个模型的详细信息
@app.route('/api/models/<int:model_id>', methods=['GET'])
@token_required
def get_model_detail(current_user, model_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取模型详细信息
    query = """
        SELECT m.*, u.username as creator_name
        FROM models m
        LEFT JOIN users u ON m.creator_id = u.id
        WHERE m.id = %s AND (m.is_system = 1 OR m.is_shared = 1 OR m.creator_id = %s)
    """
    cursor.execute(query, (model_id, current_user['id']))
    model = cursor.fetchone()

    if not model:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在或无权访问"}), 404

    # 查询用户是否收藏了此模型
    cursor.execute(
        "SELECT COUNT(*) as is_favorite FROM favorite_models WHERE user_id = %s AND model_id = %s",
        (current_user['id'], model_id)
    )
    favorite_result = cursor.fetchone()

    conn.close()

    # 处理时间格式
    if model['created_at']:
        model['created_at'] = model['created_at'].isoformat()
    if model['updated_at']:
        model['updated_at'] = model['updated_at'].isoformat()

    # 添加收藏状态
    model['is_favorite'] = favorite_result['is_favorite'] > 0 if favorite_result else False

    # 返回模型信息
    return jsonify({
        "success": True,
        "model": model
    })


# 删除模型
@app.route('/api/models/<int:model_id>', methods=['DELETE'])
@token_required
def delete_model(current_user, model_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 验证模型是否存在且属于当前用户（非系统模型）
    cursor.execute(
        "SELECT id, path, example_image, is_system FROM models WHERE id = %s AND creator_id = %s",
        (model_id, current_user['id'])
    )
    model = cursor.fetchone()

    if not model:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在或无权删除"}), 404

    if model['is_system']:
        conn.close()
        return jsonify({"success": False, "message": "系统模型不能删除"}), 403

    try:
        # 删除模型文件
        if model['path'] and os.path.exists(model['path']):
            os.remove(model['path'])

        # 删除示例图片文件
        if model['example_image'] and os.path.exists(model['example_image']):
            os.remove(model['example_image'])

        # 删除相关的收藏记录
        cursor.execute("DELETE FROM favorite_models WHERE model_id = %s", (model_id,))

        # 删除数据库中的模型记录
        cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型已成功删除"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"删除模型失败: {str(e)}"}), 500


# 视频上传接口
@app.route('/api/videos/upload', methods=['POST'])
@token_required
def upload_video(current_user):
    app.logger.info(f"接收到视频上传请求，用户ID: {current_user['id']}")

    if 'file' not in request.files:
        app.logger.warning("没有发现文件部分")
        return jsonify({"success": False, "message": "没有文件"}), 400

    file = request.files['file']
    app.logger.info(f"上传的文件: {file.filename}, 内容类型: {file.content_type}")

    if file.filename == '':
        return jsonify({"success": False, "message": "未选择文件"}), 400

    if file and allowed_file(file.filename):
        # 检查是否为视频文件
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        video_extensions = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}

        app.logger.info(f"文件扩展名: {file_ext}, 是否为视频: {file_ext in video_extensions}")

        if file_ext not in video_extensions:
            return jsonify({"success": False, "message": "请上传视频文件"}), 400

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"

        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        app.logger.info(f"文件已保存到: {filepath}")

        # 尝试获取视频元数据
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                app.logger.info(
                    f"视频元数据: 宽={width}, 高={height}, 帧率={fps}, 总帧数={frame_count}, 时长={duration}")
            else:
                app.logger.warning(f"无法打开视频文件: {filepath}")
                width, height, duration, frame_count = 0, 0, 0, 0
        except Exception as e:
            app.logger.error(f"获取视频元数据失败: {str(e)}")
            width, height, duration, frame_count = 0, 0, 0, 0

        # 保存上传记录到数据库
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            app.logger.info("开始将视频记录插入数据库")
            cursor.execute(
                """INSERT INTO images 
                   (user_id, original_filename, storage_filename, upload_path, created_at,
                    file_type, width, height, duration, frame_count, file_size) 
                   VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)""",
                (current_user['id'], filename, unique_filename, filepath, 'video',
                 width, height, duration, frame_count, os.path.getsize(filepath))
            )
            conn.commit()
            video_id = cursor.lastrowid
            app.logger.info(f"视频记录已保存，ID: {video_id}")

            conn.close()

            return jsonify({
                "success": True,
                "message": "视频上传成功",
                "video": {
                    "id": video_id,
                    "filename": filename,
                    "path": filepath,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "frame_count": frame_count
                }
            })
        except Exception as e:
            app.logger.error(f"保存视频记录失败: {str(e)}")
            conn.rollback()
            conn.close()

            # 如果数据库操作失败，删除上传的文件
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info(f"已删除临时上传文件: {filepath}")
                except:
                    app.logger.warning(f"无法删除临时上传文件: {filepath}")

            return jsonify({"success": False, "message": f"保存记录失败: {str(e)}"}), 500

    return jsonify({"success": False, "message": "文件类型不允许"}), 400


# 视频处理状态存储
video_processing_status = {}


# 视频处理接口
@app.route('/api/videos/process', methods=['POST'])
@token_required
def process_video(current_user):
    # 获取前端传来的JSON数据
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求数据无效"}), 400

    # 获取参数
    video_id = data.get('videoId')
    model_id = data.get('modelId')
    start_time = data.get('startTime', 0)
    end_time = data.get('endTime')
    remark = data.get('remark', '')

    if not video_id or not model_id:
        return jsonify({"success": False, "message": "缺少必要参数"}), 400

    # 连接数据库
    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取视频信息 (在images表中，file_type为video的记录)
    cursor.execute("""
        SELECT * FROM images 
        WHERE id = %s AND user_id = %s AND file_type = 'video'
    """, (video_id, current_user['id']))
    video_data = cursor.fetchone()

    if not video_data:
        conn.close()
        return jsonify({"success": False, "message": "找不到指定的视频"}), 404

    # 检查视频文件是否存在
    video_path = video_data['upload_path']
    if not os.path.exists(video_path):
        conn.close()
        return jsonify({"success": False, "message": "视频文件不存在"}), 404

    # 检查是否为视频文件
    file_ext = os.path.splitext(video_data['original_filename'])[1].lower() if '.' in video_data[
        'original_filename'] else ''
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'}

    if file_ext not in video_extensions:
        conn.close()
        return jsonify({"success": False, "message": "所选文件不是视频"}), 400

    # 获取模型信息
    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()

    if not model_data:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在"}), 404

    # 检查模型是否支持视频
    if not model_data.get('supports_video', True):
        conn.close()
        return jsonify({"success": False, "message": "所选模型不支持视频处理"}), 400

    # 创建处理历史记录
    try:
        # 生成结果文件名
        result_filename = f"video_result_{uuid.uuid4().hex}.mp4"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # 记录处理开始
        app.logger.info(f"开始处理视频 - ID: {video_id}, 模型: {model_data['name']}")
        start_processing_time = time.time()

        # 直接在当前请求中处理视频，而不是启动后台线程
        try:
            # 加载视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                app.logger.error(f"无法打开视频文件: {video_path}")
                raise ValueError("无法打开视频文件")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            app.logger.info(f"视频信息 - 宽度: {width}, 高度: {height}, 帧率: {fps}, 总帧数: {total_frames}")

            # 转换开始和结束时间到帧数
            start_frame = int(start_time * fps) if start_time else 0
            end_frame = int(end_time * fps) if end_time else total_frames

            # 设置起始帧
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 计算总处理帧数
            frames_to_process = min(end_frame, total_frames) - start_frame
            app.logger.info(f"处理范围 - 起始帧: {start_frame}, 结束帧: {end_frame}, 总处理帧数: {frames_to_process}")

            # 获取原视频编码器和扩展名
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = (
                chr((fourcc_int & 0xFF)),
                chr((fourcc_int >> 8) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 24) & 0xFF)
            )
            fourcc_str = ''.join(fourcc)
            allowed_fourcc = ['mp4v', 'xvid', 'avc1', 'h264', 'h265', 'hevc', 'mjpg']
            if fourcc_str.strip() == '' or fourcc_str.lower() not in allowed_fourcc:
                fourcc_str = 'mp4v'
            file_ext = os.path.splitext(video_data['original_filename'])[1].lower()
            result_filename = f"video_result_{uuid.uuid4().hex}{file_ext}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
            out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))

            # 加载YOLO模型
            app.logger.info(f"开始加载YOLO模型: {model_data['path']}")

            # 检查模型文件是否存在
            if not os.path.exists(model_data['path']):
                raise FileNotFoundError(f"模型文件不存在: {model_data['path']}")

            # 直接使用YOLO加载模型
            model = YOLO(model_data['path'])
            app.logger.info(f"YOLO模型加载成功: {type(model).__name__}")

            # 为每个类别分配不同颜色
            colors = {}
            processed_frames = 0
            detections_sum = 0
            all_categories = set()
            conf_sum = 0.0
            detection_count = 0

            # 设置跳帧参数 - 提高处理性能
            process_every_n_frames = 2  # 每隔n帧处理一次

            current_frame = start_frame
            last_detections = None

            # 设置检测处理函数
            def detect_objects(frame):
                # 直接使用YOLO模型的predict方法
                results = model.predict(frame, conf=0.25, verbose=False)

                # 将检测结果转换为DataFrame格式，以便与后续处理兼容
                detections_list = []

                # 只处理第一个结果（因为只有一帧）
                if len(results) > 0:
                    result = results[0]

                    # 获取边界框、置信度和类别
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # 获取边界框坐标
                        box = boxes[i].xyxy.cpu().numpy()[0]  # 转换为xyxy格式并获取numpy数组
                        x1, y1, x2, y2 = box

                        # 获取置信度和类别
                        conf = float(boxes[i].conf.cpu().numpy()[0])
                        cls = int(boxes[i].cls.cpu().numpy()[0])
                        cls_name = result.names[cls]

                        detections_list.append({
                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2,
                            'confidence': conf,
                            'class': cls,
                            'name': cls_name
                        })

                # 创建DataFrame
                df_detections = pd.DataFrame(detections_list)
                return df_detections, len(detections_list)

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    app.logger.warning(f"读取第{current_frame}帧失败，提前结束")
                    break

                # 是否需要处理这一帧
                should_process = processed_frames % process_every_n_frames == 0

                if should_process:
                    try:
                        # 使用模型进行目标检测
                        detections, num_detections = detect_objects(frame)
                        last_detections = detections

                        # 更新统计信息
                        detections_sum += num_detections

                        # 在帧上绘制检测结果
                        for _, detection in detections.iterrows():
                            # 提取检测信息
                            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(
                                detection['xmax']), int(
                                detection['ymax'])
                            class_name = detection['name']
                            confidence = detection['confidence']

                            # 更新统计数据
                            conf_sum += confidence
                            detection_count += 1
                            all_categories.add(class_name)

                            # 为新类别分配颜色
                            if class_name not in colors:
                                colors[class_name] = (
                                    np.random.randint(0, 255),
                                    np.random.randint(0, 255),
                                    np.random.randint(0, 255)
                                )

                            # 绘制边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_name], 2)

                            # 添加标签
                            label = f"{class_name}: {confidence:.2f}"
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                            # 绘制标签背景
                            cv2.rectangle(
                                frame,
                                (x1, y1 - label_height - 10),
                                (x1 + label_width, y1),
                                colors[class_name],
                                -1
                            )

                            # 绘制标签文本
                            cv2.putText(
                                frame,
                                label,
                                (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )
                    except Exception as frame_error:
                        app.logger.error(f"处理第 {current_frame} 帧时出错: {str(frame_error)}")
                        # 跳过此帧继续处理
                elif last_detections is not None and len(last_detections) > 0:
                    # 使用最后一次的检测结果绘制 (重用前一帧的结果以减少计算)
                    for _, detection in last_detections.iterrows():
                        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(
                            detection['ymax'])
                        class_name = detection['name']

                        # 绘制边界框和标签
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(class_name, (0, 255, 0)), 2)

                        # 添加标签
                        label = f"{class_name}: {detection['confidence']:.2f}"
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                        # 绘制标签背景
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width, y1),
                            colors.get(class_name, (0, 255, 0)),
                            -1
                        )

                        # 绘制标签文本
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                # 将处理后的帧写入输出视频
                out.write(frame)

                # 更新进度
                processed_frames += 1
                current_frame += 1

            # 释放资源
            cap.release()
            out.release()

            # 计算处理时间和平均精度
            processing_time = time.time() - start_processing_time
            avg_confidence = conf_sum / detection_count if detection_count > 0 else 0
            avg_accuracy = avg_confidence * 100  # 转换为百分比作为精度

            app.logger.info(
                f"视频处理完成 - 处理时间: {processing_time:.2f}秒, 识别对象: {detections_sum}, 平均置信度: {avg_confidence:.4f}")

            # 检查生成的视频文件是否有效
            if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                app.logger.error(f"视频处理完成但结果文件无效: {result_path}")
                raise ValueError("处理失败：无法生成有效的视频结果")

            # 复制文件到静态目录以便前端访问
            static_video_filename = f"result_{uuid.uuid4().hex}.mp4"
            static_video_path = os.path.join('static', 'videos', static_video_filename)
            abs_static_path = os.path.join(os.getcwd(), static_video_path)

            # 确保静态视频目录存在
            os.makedirs(os.path.dirname(abs_static_path), exist_ok=True)

            # 复制文件
            shutil.copy2(result_path, abs_static_path)
            app.logger.info(f"已复制视频文件到静态目录: {abs_static_path}")

            # 修复视频文件权限确保web服务器可以访问
            try:
                # 在Linux/Unix系统上设置权限
                import stat
                os.chmod(abs_static_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            except Exception as perms_err:
                app.logger.warning(f"设置视频文件权限失败(可忽略): {str(perms_err)}")

            # 更新处理状态和数据库
            categories_json = json.dumps(list(all_categories))

            # 记录处理历史
            result_file_size = os.path.getsize(result_path)
            cursor.execute(
                """INSERT INTO processing_history 
                   (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, 
                    file_type, status, progress, categories, object_count, data_size) 
                   VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)""",
                (current_user['id'], video_id, model_id, result_path, avg_accuracy, processing_time,
                 'video', 'completed', 100, categories_json, detections_sum, result_file_size)
            )
            conn.commit()
            history_id = cursor.lastrowid

            # 更新模型使用次数
            cursor.execute("UPDATE models SET usage_count = usage_count + 1 WHERE id = %s", (model_id,))
            conn.commit()

            # 返回结果
            static_url = f"/static/videos/{static_video_filename}"
            result_url = f"/api/videos/preview/{history_id}?token={request.headers.get('Authorization').split(' ')[1]}"

            conn.close()

            return jsonify({
                "success": True,
                "message": "视频处理成功",
                "result": {
                    "id": history_id,
                    "status": "completed",
                    "accuracy": avg_accuracy,
                    "processingTime": processing_time,
                    "resultUrl": result_url,
                    "staticUrl": static_url,
                    "videoFilename": static_video_filename,
                    "categories": list(all_categories),
                    "objectCount": detections_sum,
                    "contentType": "video/mp4"
                }
            })

        except Exception as process_error:
            conn.rollback()
            app.logger.error(f"视频处理失败: {str(process_error)}", exc_info=True)
            # 记录错误状态
            cursor.execute(
                """INSERT INTO processing_history 
                   (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, 
                    file_type, status, progress, categories, object_count, data_size) 
                   VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)""",
                (current_user['id'], video_id, model_id, "", 0, 0,
                 'video', 'error', 0, json.dumps([]), 0, 0)
            )
            conn.commit()
            conn.close()
            return jsonify({"success": False, "message": f"处理失败: {str(process_error)}"}), 500

    except Exception as e:
        app.logger.error(f"启动视频处理失败: {str(e)}")
        if conn:
            conn.rollback()
            conn.close()
        return jsonify({"success": False, "message": f"处理失败: {str(e)}"}), 500


@app.route('/api/videos/status/<int:history_id>', methods=['GET'])
@token_required
def get_video_processing_status(current_user, history_id):
    # 检查历史记录是否存在且属于当前用户
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取处理记录
        query = """
        SELECT h.id, h.status, h.progress, h.processing_time, h.accuracy, 
               h.result_path, h.object_count, h.categories,
               i.upload_path as video_path, i.original_filename, 
               m.name as model_name
        FROM processing_history h
        JOIN images i ON h.image_id = i.id
        JOIN models m ON h.model_id = m.id
        WHERE h.id = %s AND h.user_id = %s AND h.file_type = 'video'
        """
        cursor.execute(query, (history_id, current_user['id']))
        record = cursor.fetchone()

        if not record:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': '找不到指定的处理记录或您无权访问'
            }), 404

        # 构建结果URL
        result_url = None
        static_url = None
        video_filename = None

        app.logger.info(f"视频状态请求 - 历史ID: {history_id}, 状态: {record['status']}, 进度: {record['progress']}%")

        # 无论状态如何，都尝试创建静态URL(减少前端轮询误判情况)
        if record['result_path'] and os.path.exists(record['result_path']):
            auth_header = request.headers.get('Authorization')
            token = auth_header.split(' ')[1] if auth_header and ' ' in auth_header else ''
            result_url = f"/api/videos/preview/{record['id']}?token={token}"

            # 为视频文件创建一个唯一的静态访问链接
            try:
                original_path = record['result_path']
                video_filename = f"result_{record['id']}.mp4"
                static_video_path = os.path.join('static', 'videos', video_filename)
                abs_static_path = os.path.join(os.getcwd(), static_video_path)

                # 确保静态视频目录存在
                os.makedirs(os.path.dirname(abs_static_path), exist_ok=True)

                # 检查文件是否存在且有效
                if os.path.getsize(original_path) > 0:
                    # 如果文件不存在或需要更新，则复制文件
                    if not os.path.exists(abs_static_path) or \
                            os.path.getmtime(abs_static_path) != os.path.getmtime(original_path) or \
                            os.path.getsize(abs_static_path) != os.path.getsize(original_path):
                        try:
                            # 复制文件
                            shutil.copy2(original_path, abs_static_path)
                            app.logger.info(f"已复制视频文件到静态目录: {abs_static_path}")

                            # 修复视频文件权限确保web服务器可以访问
                            try:
                                # 在Linux/Unix系统上设置权限
                                import stat
                                os.chmod(abs_static_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                            except Exception as perms_err:
                                app.logger.warning(f"设置视频文件权限失败(可忽略): {str(perms_err)}")
                        except Exception as e:
                            app.logger.error(f"复制视频文件失败: {str(e)}")

                    # 检查静态文件是否成功创建
                    if os.path.exists(abs_static_path) and os.path.getsize(abs_static_path) > 0:
                        # 生成相对URL
                        static_url = f"/static/videos/{video_filename}"
                        app.logger.info(f"生成静态视频URL: {static_url}")
                    else:
                        app.logger.error(f"静态视频文件无效或大小为零: {abs_static_path}")
                else:
                    app.logger.error(f"原始视频文件无效或大小为零: {original_path}")
            except Exception as e:
                app.logger.error(f"处理视频路径出错: {str(e)}")
        else:
            app.logger.warning(f"视频结果文件不存在: {record.get('result_path', '未指定路径')}")

        # 解析分类信息
        categories = []
        if record['categories']:
            try:
                categories = json.loads(record['categories'])
            except:
                # 如果JSON解析失败，尝试分割字符串
                categories = record['categories'].split(',') if isinstance(record['categories'], str) else []

        # 获取状态消息
        status_message = "处理中..."
        if record['status'] == 'completed':
            status_message = "处理完成"
        elif record['status'] == 'error':
            status_message = "处理失败"

        # 准备响应数据
        processing_data = {
            'status': record['status'],
            'progress': record['progress'] or 0,
            'message': status_message,
            'processingTime': float(record['processing_time']) if record['processing_time'] is not None else None,
            'accuracy': float(record['accuracy']) if record['accuracy'] is not None else None,
            'resultUrl': result_url,
            'staticUrl': static_url,  # 添加静态文件URL
            'videoFilename': video_filename,  # 添加视频文件名
            'categories': categories,
            'objectCount': record['object_count'] or 0,
            'contentType': 'video/mp4'  # 添加内容类型
        }

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'processing': processing_data
        })
    except Exception as e:
        app.logger.error(f"获取视频处理状态出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': '获取处理状态失败: ' + str(e)
        }), 500


# 视频结果预览
@app.route('/api/videos/preview/<int:history_id>', methods=['GET'])
def preview_video_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')

    if not token:
        return jsonify({"success": False, "message": "缺少Token"}), 401

    try:
        # 验证token
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user_id = payload['sub']

        # 获取处理结果视频路径
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT h.result_path, h.status 
               FROM processing_history h 
               WHERE h.id = %s AND h.user_id = %s AND h.file_type = 'video'""",
            (history_id, user_id)
        )
        record = cursor.fetchone()
        conn.close()

        if not record:
            return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

        # 检查处理状态
        if record['status'] != 'completed':
            return jsonify({"success": False, "message": "视频处理尚未完成", "status": record['status']}), 400

        result_path = record['result_path']

        # 检查结果文件是否存在
        if not os.path.exists(result_path):
            app.logger.error(f"视频结果文件不存在: {result_path}")
            return jsonify({"success": False, "message": "找不到视频结果文件"}), 404

        # 检查文件大小
        if os.path.getsize(result_path) == 0:
            app.logger.error(f"视频结果文件大小为零: {result_path}")
            return jsonify({"success": False, "message": "视频结果文件无效"}), 400

        # 设置正确的缓存控制头和范围请求支持
        response = send_file(
            result_path,
            mimetype='video/mp4',
            conditional=True  # 启用If-Range/Range请求支持
        )

        # 添加必要的头信息
        response.headers['Accept-Ranges'] = 'bytes'  # 支持范围请求
        response.headers['Access-Control-Allow-Origin'] = '*'  # CORS支持

        # 设置缓存控制
        response.headers['Cache-Control'] = 'public, max-age=3600'  # 缓存一小时
        app.logger.info(f"返回视频预览文件: {result_path}")

        return response

    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "message": "Token已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "message": "Token无效"}), 401
    except Exception as e:
        app.logger.error(f"获取视频预览失败: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"获取视频失败: {str(e)}"}), 500


# 下载视频处理结果
@app.route('/api/videos/download/<int:history_id>', methods=['GET'])
def download_video_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')

    if not token:
        return jsonify({"success": False, "message": "缺少Token"}), 401

    try:
        # 验证token
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user_id = payload['sub']

        # 检查用户是否存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return jsonify({"success": False, "message": "用户不存在"}), 401

        # 获取处理结果信息
        cursor.execute(
            """SELECT h.*, i.original_filename 
               FROM processing_history h 
               JOIN images i ON h.image_id = i.id 
               WHERE h.id = %s AND h.user_id = %s""",
            (history_id, user_id)
        )
        history = cursor.fetchone()
        conn.close()

        if not history:
            return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

        # 设置下载的文件名：原始文件名基础上加上_result后缀
        original_name = history['original_filename']
        name_parts = original_name.rsplit('.', 1)
        download_name = f"{name_parts[0]}_result.mp4" if len(name_parts) > 1 else f"{original_name}_result.mp4"

        return send_file(
            history['result_path'],
            as_attachment=True,
            download_name=download_name
        )
    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "message": "Token已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "message": "Token无效"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": f"下载失败: {str(e)}"}), 500


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """
    检查系统状态，包括视频处理功能是否正常工作
    """
    status = {
        "success": True,
        "status": "online",
        "features": {
            "images": {
                "upload": True,
                "process": True
            },
            "videos": {
                "upload": True,
                "process": True
            },
            "models": {
                "detection": True
            }
        },
        "version": "1.0.0",
        "serverTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 检查视频目录和模型目录是否存在
    upload_dir = os.path.join('static', 'uploads')
    if not os.path.exists(upload_dir):
        try:
            os.makedirs(upload_dir)
            app.logger.info(f"创建上传目录: {upload_dir}")
        except Exception as e:
            app.logger.error(f"创建上传目录失败: {str(e)}")
            status["features"]["videos"]["upload"] = False
            status["features"]["images"]["upload"] = False

    # 检查模型目录
    models_dir = os.path.join('static', 'models')
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
            app.logger.info(f"创建模型目录: {models_dir}")
        except Exception as e:
            app.logger.error(f"创建模型目录失败: {str(e)}")
            status["features"]["models"]["detection"] = False

    # 检查结果目录
    results_dir = os.path.join('static', 'results')
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            app.logger.info(f"创建结果目录: {results_dir}")
        except Exception as e:
            app.logger.error(f"创建结果目录失败: {str(e)}")
            status["features"]["videos"]["process"] = False
            status["features"]["images"]["process"] = False

    return jsonify(status)


# 专门用于静态视频文件的直接访问路由
@app.route('/api/static/video/<string:video_filename>', methods=['GET'])
def get_static_video(video_filename):
    """提供对静态视频文件的直接访问"""
    try:
        # 安全检查：确保文件名不包含目录遍历等危险字符
        if '..' in video_filename or '/' in video_filename:
            return jsonify({"success": False, "message": "无效的文件名"}), 400

        # 构建视频文件路径
        video_path = os.path.join(os.getcwd(), 'static', 'videos', video_filename)
        app.logger.info(f"请求访问静态视频: {video_path}")

        # 检查文件是否存在
        if not os.path.exists(video_path):
            app.logger.error(f"静态视频文件不存在: {video_path}")

            # 尝试从结果文件夹复制
            try:
                # 从文件名中提取历史ID（假设文件名格式为 result_X.mp4）
                if video_filename.startswith("result_") and video_filename.endswith(".mp4"):
                    history_id = int(video_filename[7:-4])  # 提取数字部分

                    # 查询数据库获取原始结果路径
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT result_path FROM processing_history WHERE id = %s AND file_type = 'video'",
                        (history_id,)
                    )
                    record = cursor.fetchone()
                    cursor.close()
                    conn.close()

                    if record and record['result_path'] and os.path.exists(record['result_path']):
                        original_path = record['result_path']
                        # 确保目录存在
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        # 复制文件
                        shutil.copy2(original_path, video_path)
                        app.logger.info(f"从结果目录复制视频文件: {original_path} -> {video_path}")
                    else:
                        app.logger.error(f"找不到原始结果文件，历史ID: {history_id}")
                        return jsonify({"success": False, "message": "视频文件不存在"}), 404
                else:
                    return jsonify({"success": False, "message": "视频文件不存在"}), 404
            except Exception as copy_err:
                app.logger.error(f"尝试复制视频文件失败: {str(copy_err)}")
                return jsonify({"success": False, "message": "视频文件不存在"}), 404

        # 检查文件大小
        if os.path.getsize(video_path) == 0:
            app.logger.error(f"视频文件大小为零: {video_path}")
            return jsonify({"success": False, "message": "视频文件无效"}), 400

        # 返回视频文件
        response = send_file(
            video_path,
            mimetype='video/mp4'
        )
        # 添加必要的CORS和缓存控制头
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cache-Control'] = 'public, max-age=86400'  # 缓存一天
        return response
    except Exception as e:
        app.logger.error(f"提供静态视频失败: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"获取视频失败: {str(e)}"}), 500


# 管理员API：获取所有权限申请
@app.route('/api/admin/permissions', methods=['GET'])
@token_required
def admin_get_all_permissions(current_user):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取参数
    status = request.args.get('status', 'all')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    offset = (page - 1) * per_page

    # 构建查询
    query_conditions = []
    params = []

    if status != 'all':
        query_conditions.append("pr.status = %s")
        params.append(status)

    where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"

    # 查询总数
    cursor.execute(
        f"""SELECT COUNT(*) as total 
           FROM permission_requests pr 
           WHERE {where_clause}"""
        , params)
    total = cursor.fetchone()['total']

    # 执行查询
    cursor.execute(
        f"""SELECT pr.*, u.username, u.email, u.name, u.organization
           FROM permission_requests pr
           JOIN users u ON pr.user_id = u.id
           WHERE {where_clause}
           ORDER BY pr.created_at DESC
           LIMIT %s OFFSET %s""",
        params + [per_page, offset]
    )
    permissions = cursor.fetchall()
    conn.close()

    # 处理日期格式和额外信息
    for permission in permissions:
        permission['created_at'] = permission['created_at'].isoformat() if hasattr(permission['created_at'],
                                                                                   'isoformat') else str(
            permission['created_at'])
        permission['updated_at'] = permission['updated_at'].isoformat() if hasattr(permission['updated_at'],
                                                                                   'isoformat') else str(
            permission['updated_at'])

        if permission['additional_info']:
            try:
                permission['additional_info'] = json.loads(permission['additional_info'])
            except:
                pass

    return jsonify({
        "success": True,
        "permissions": permissions,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    })


# 管理员API：审批权限申请
@app.route('/api/admin/permissions/<int:request_id>/approve', methods=['POST'])
@token_required
def admin_approve_permission(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权操作，需要管理员权限"}), 403

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取申请信息
    cursor.execute(
        """SELECT pr.*, u.username, u.email 
           FROM permission_requests pr
           JOIN users u ON pr.user_id = u.id
           WHERE pr.id = %s""",
        (request_id,)
    )
    permission = cursor.fetchone()

    if not permission:
        conn.close()
        return jsonify({"success": False, "message": "申请不存在"}), 404

    if permission['status'] != 'pending':
        conn.close()
        return jsonify({"success": False, "message": "只能审批待处理的申请"}), 400

    try:
        # 更新申请状态
        cursor.execute(
            """UPDATE permission_requests 
               SET status = 'approved', updated_at = NOW() 
               WHERE id = %s""",
            (request_id,)
        )

        # 创建通知
        cursor.execute(
            """INSERT INTO notifications 
               (user_id, title, content, type, is_read, created_at) 
               VALUES (%s, %s, %s, %s, 0, NOW())""",
            (permission['user_id'], "权限申请已批准",
             f"您的{get_request_type_name(permission['request_type'])}申请已获批准，现在可以使用相关功能了。", "success")
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "权限申请已批准"
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"操作失败: {str(e)}"}), 500


# 管理员API：拒绝权限申请
@app.route('/api/admin/permissions/<int:request_id>/reject', methods=['POST'])
@token_required
def admin_reject_permission(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权操作，需要管理员权限"}), 403

    # 获取拒绝原因
    data = request.json
    reject_reason = data.get('reason', '未提供拒绝原因')

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取申请信息
    cursor.execute(
        """SELECT pr.*, u.username, u.email 
           FROM permission_requests pr
           JOIN users u ON pr.user_id = u.id
           WHERE pr.id = %s""",
        (request_id,)
    )
    permission = cursor.fetchone()

    if not permission:
        conn.close()
        return jsonify({"success": False, "message": "申请不存在"}), 404

    if permission['status'] != 'pending':
        conn.close()
        return jsonify({"success": False, "message": "只能拒绝待处理的申请"}), 400

    try:
        # 更新申请状态
        cursor.execute(
            """UPDATE permission_requests 
               SET status = 'rejected', updated_at = NOW() 
               WHERE id = %s""",
            (request_id,)
        )

        # 创建通知
        cursor.execute(
            """INSERT INTO notifications 
               (user_id, title, content, type, is_read, created_at) 
               VALUES (%s, %s, %s, %s, 0, NOW())""",
            (permission['user_id'], "权限申请被拒绝",
             f"您的{get_request_type_name(permission['request_type'])}申请被拒绝。原因：{reject_reason}", "error")
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "已拒绝权限申请"
        })

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"操作失败: {str(e)}"}), 500


# 管理员API：获取权限申请详情
@app.route('/api/admin/permissions/<int:request_id>', methods=['GET'])
@token_required
def admin_get_permission_detail(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取权限申请详情
    cursor.execute(
        """SELECT pr.*, u.username, u.email, u.name, u.organization, u.department
           FROM permission_requests pr
           JOIN users u ON pr.user_id = u.id
           WHERE pr.id = %s""",
        (request_id,)
    )
    permission = cursor.fetchone()
    conn.close()

    if not permission:
        return jsonify({"success": False, "message": "申请不存在"}), 404

    # 处理日期格式
    permission['created_at'] = permission['created_at'].isoformat() if hasattr(permission['created_at'],
                                                                               'isoformat') else str(
        permission['created_at'])
    permission['updated_at'] = permission['updated_at'].isoformat() if hasattr(permission['updated_at'],
                                                                               'isoformat') else str(
        permission['updated_at'])

    # 解析额外信息
    if permission['additional_info']:
        try:
            permission['additional_info'] = json.loads(permission['additional_info'])
        except:
            pass

    return jsonify({
        "success": True,
        "permission": permission
    })


# 批量视频处理接口
@app.route('/api/videos/batch-process', methods=['POST'])
@token_required
def batch_process_videos(current_user):
    app.logger.info("开始批量视频处理请求")
    # 获取前端传来的JSON数据
    data = request.get_json()
    if not data:
        app.logger.error("批量处理请求数据无效")
        return jsonify({"success": False, "message": "请求数据无效"}), 400

    # 获取参数
    video_ids = data.get('videoIds')
    model_id = data.get('modelId')
    start_time = data.get('startTime', 0)
    end_time = data.get('endTime')

    app.logger.info(f"批量处理参数: 视频IDs={video_ids}, 模型ID={model_id}, 开始时间={start_time}, 结束时间={end_time}")

    if not video_ids or not model_id or not isinstance(video_ids, list) or len(video_ids) == 0:
        app.logger.error("批量处理缺少必要参数或视频ID列表为空")
        return jsonify({"success": False, "message": "缺少必要参数或视频ID列表为空"}), 400

    # 连接数据库
    conn = get_db_connection()
    cursor = conn.cursor()

    # 获取模型信息
    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()

    if not model_data:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在"}), 404

    # 检查模型文件是否存在
    if not os.path.exists(model_data['path']):
        app.logger.error(f"模型文件不存在: {model_data['path']}")
        conn.close()
        return jsonify({"success": False, "message": f"模型文件不存在: {model_data['path']}"}), 404

    # 尝试预加载模型验证是否可以正常工作
    try:
        app.logger.info(f"预加载模型以验证: {model_data['path']}")
        model = YOLO(model_data['path'])
        app.logger.info(f"模型预加载成功: {type(model).__name__}")
    except Exception as model_err:
        app.logger.error(f"模型加载失败: {str(model_err)}")
        conn.close()
        return jsonify({"success": False, "message": f"模型加载失败: {str(model_err)}"}), 500

    # 处理结果
    results = []
    total_success = 0
    total_count = len(video_ids)

    # 准备一个共享的模型实例处理所有视频
    model = YOLO(model_data['path'])

    # 处理每个视频
    for video_id in video_ids:
        try:
            # 获取视频信息
            cursor.execute("""
                SELECT * FROM images 
                WHERE id = %s AND user_id = %s AND file_type = 'video'
            """, (video_id, current_user['id']))
            video_data = cursor.fetchone()

            if not video_data:
                results.append({
                    "videoId": video_id,
                    "success": False,
                    "message": "视频不存在或无权访问"
                })
                continue

            # 检查视频文件是否存在
            video_path = video_data['upload_path']
            if not os.path.exists(video_path):
                results.append({
                    "videoId": video_id,
                    "success": False,
                    "message": "视频文件不存在"
                })
                continue

            # 测试打开视频文件
            try:
                test_cap = cv2.VideoCapture(video_path)
                if not test_cap.isOpened():
                    results.append({
                        "videoId": video_id,
                        "success": False,
                        "message": "无法打开视频文件"
                    })
                    test_cap.release()
                    continue
                test_cap.release()
            except Exception as vid_err:
                results.append({
                    "videoId": video_id,
                    "success": False,
                    "message": f"视频文件测试失败: {str(vid_err)}"
                })
                continue

            # 创建结果文件名和路径
            result_filename = f"video_result_{uuid.uuid4().hex}.mp4"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            # 开始处理视频
            app.logger.info(f"开始处理视频 ID: {video_id}, 路径: {video_path}")
            start_processing_time = time.time()

            try:
                # 加载视频
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 转换开始和结束时间到帧数
                start_frame = int(start_time * fps) if start_time else 0
                end_frame = int(end_time * fps) if end_time else total_frames

                # 设置起始帧
                if start_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # 获取原视频编码器和扩展名
                # 获取原视频编码器和扩展名
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = (
                    chr((fourcc_int & 0xFF)),
                    chr((fourcc_int >> 8) & 0xFF),
                    chr((fourcc_int >> 16) & 0xFF),
                    chr((fourcc_int >> 24) & 0xFF)
                )
                fourcc_str = ''.join(fourcc)
                allowed_fourcc = ['mp4v', 'xvid', 'avc1', 'h264', 'h265', 'hevc', 'mjpg']
                if fourcc_str.strip() == '' or fourcc_str.lower() not in allowed_fourcc:
                    fourcc_str = 'mp4v'
                file_ext = os.path.splitext(video_data['original_filename'])[1].lower()
                result_filename = f"video_result_{uuid.uuid4().hex}{file_ext}"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
                out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))

                # 设置处理参数
                colors = {}
                processed_frames = 0
                detections_sum = 0
                all_categories = set()
                conf_sum = 0.0
                detection_count = 0
                process_every_n_frames = 2  # 每隔n帧处理一次

                current_frame = start_frame
                last_detections = None
                frames_to_process = min(end_frame, total_frames) - start_frame

                # 定义检测函数
                def batch_detect_objects(frame):
                    # 使用YOLO模型进行预测
                    results = model.predict(frame, conf=0.25, verbose=False)

                    # 将检测结果转换为列表格式
                    detections_list = []

                    if len(results) > 0:
                        result = results[0]
                        boxes = result.boxes

                        for i in range(len(boxes)):
                            box = boxes[i].xyxy.cpu().numpy()[0]
                            x1, y1, x2, y2 = box
                            conf = float(boxes[i].conf.cpu().numpy()[0])
                            cls = int(boxes[i].cls.cpu().numpy()[0])
                            cls_name = result.names[cls]

                            detections_list.append({
                                'xmin': x1,
                                'ymin': y1,
                                'xmax': x2,
                                'ymax': y2,
                                'confidence': conf,
                                'class': cls,
                                'name': cls_name
                            })

                    # 创建DataFrame
                    df_detections = pd.DataFrame(detections_list)
                    return df_detections, len(detections_list)

                # 处理每一帧
                while current_frame < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 是否需要处理这一帧
                    should_process = processed_frames % process_every_n_frames == 0

                    if should_process:
                        try:
                            # 使用模型进行目标检测
                            detections, num_detections = batch_detect_objects(frame)
                            last_detections = detections

                            # 更新统计信息
                            detections_sum += num_detections

                            # 在帧上绘制检测结果
                            for _, detection in detections.iterrows():
                                # 提取检测信息
                                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(
                                    detection['xmax']), int(detection['ymax'])
                                class_name = detection['name']
                                confidence = detection['confidence']

                                # 更新统计数据
                                conf_sum += confidence
                                detection_count += 1
                                all_categories.add(class_name)

                                # 为新类别分配颜色
                                if class_name not in colors:
                                    colors[class_name] = (
                                        np.random.randint(0, 255),
                                        np.random.randint(0, 255),
                                        np.random.randint(0, 255)
                                    )

                                # 绘制边界框
                                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_name], 2)

                                # 添加标签
                                label = f"{class_name}: {confidence:.2f}"
                                (label_width, label_height), baseline = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                                # 绘制标签背景
                                cv2.rectangle(
                                    frame,
                                    (x1, y1 - label_height - 10),
                                    (x1 + label_width, y1),
                                    colors[class_name],
                                    -1
                                )

                                # 绘制标签文本
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA
                                )
                        except Exception as frame_error:
                            app.logger.error(f"处理第 {current_frame} 帧时出错: {str(frame_error)}")
                            # 跳过出错帧继续处理
                    elif last_detections is not None and len(last_detections) > 0:
                        # 使用最后一次的检测结果绘制
                        for _, detection in last_detections.iterrows():
                            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(
                                detection['xmax']), int(detection['ymax'])
                            class_name = detection['name']

                            # 绘制边界框和标签
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(class_name, (0, 255, 0)), 2)

                            # 添加标签
                            label = f"{class_name}: {detection['confidence']:.2f}"
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                            # 绘制标签背景
                            cv2.rectangle(
                                frame,
                                (x1, y1 - label_height - 10),
                                (x1 + label_width, y1),
                                colors.get(class_name, (0, 255, 0)),
                                -1
                            )

                            # 绘制标签文本
                            cv2.putText(
                                frame,
                                label,
                                (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                    # 将处理后的帧写入输出视频
                    out.write(frame)

                    # 更新进度
                    processed_frames += 1
                    current_frame += 1

                # 释放资源
                cap.release()
                out.release()

                # 计算处理时间和平均精度
                processing_time = time.time() - start_processing_time
                avg_confidence = conf_sum / detection_count if detection_count > 0 else 0
                avg_accuracy = avg_confidence * 100  # 转换为百分比

                # 检查生成的视频文件是否有效
                if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                    raise ValueError("处理失败：无法生成有效的视频结果")

                # 复制文件到静态目录以便前端访问
                static_video_filename = f"result_{uuid.uuid4().hex}.mp4"
                static_video_path = os.path.join('static', 'videos', static_video_filename)
                abs_static_path = os.path.join(os.getcwd(), static_video_path)

                # 确保静态视频目录存在
                os.makedirs(os.path.dirname(abs_static_path), exist_ok=True)

                # 复制文件
                shutil.copy2(result_path, abs_static_path)

                # 更新数据库
                categories_json = json.dumps(list(all_categories))

                # 记录处理历史
                result_file_size = os.path.getsize(result_path)
                cursor.execute(
                    """INSERT INTO processing_history 
                       (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, 
                        file_type, status, progress, categories, object_count, data_size) 
                       VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)""",
                    (current_user['id'], video_id, model_id, result_path, avg_accuracy, processing_time,
                     'video', 'completed', 100, categories_json, detections_sum, result_file_size)
                )
                conn.commit()
                history_id = cursor.lastrowid

                # 返回结果
                static_url = f"/static/videos/{static_video_filename}"
                result_url = f"/api/videos/preview/{history_id}?token={request.headers.get('Authorization').split(' ')[1]}"

                # 添加结果到列表
                results.append({
                    "videoId": video_id,
                    "success": True,
                    "original_filename": video_data['original_filename'],
                    "result": {
                        "id": history_id,
                        "status": "completed",
                        "accuracy": avg_accuracy,
                        "processingTime": processing_time,
                        "resultUrl": result_url,
                        "staticUrl": static_url,
                        "videoFilename": static_video_filename,
                        "categories": list(all_categories),
                        "detectedObjects": detections_sum,
                        "contentType": "video/mp4"
                    }
                })

                total_success += 1

            except Exception as process_error:
                app.logger.error(f"视频 {video_id} 处理失败: {str(process_error)}", exc_info=True)
                results.append({
                    "videoId": video_id,
                    "success": False,
                    "original_filename": video_data.get('original_filename', ''),
                    "message": f"处理失败: {str(process_error)}"
                })

        except Exception as video_error:
            app.logger.error(f"处理视频 {video_id} 时发生异常: {str(video_error)}")
            results.append({
                "videoId": video_id,
                "success": False,
                "message": f"处理失败: {str(video_error)}"
            })

    # 更新模型使用次数 - 只增加一次，不管处理了多少视频
    cursor.execute("UPDATE models SET usage_count = usage_count + 1 WHERE id = %s", (model_id,))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": f"批量处理完成，成功处理 {total_success} 个视频",
        "results": results,
        "successCount": total_success,
        "totalCount": total_count
    })


# 实时监测API
# 开始实时监测
@app.route('/api/realtime/start', methods=['POST'])
@token_required
def start_realtime_monitoring(current_user):
    app.logger.info(f"开始实时监测 - 用户ID: {current_user['id']}")
    data = request.json

    if not data:
        return jsonify({"success": False, "message": "缺少必要参数"}), 400

    model_id = data.get('modelId')
    record_video = data.get('recordVideo', False)  # 是否录制视频
    remark = data.get('remark', '')

    # 获取前端传来的参数
    detection_threshold = float(data.get('detection_threshold', 0.25))  # 检测阈值
    detection_frequency = int(data.get('detection_frequency', 5))  # 检测频率(帧)
    save_frequency = int(data.get('save_frequency', 10))  # 保存频率(秒)

    # 确保参数在合理范围内
    detection_threshold = max(0.1, min(0.9, detection_threshold))
    detection_frequency = max(1, min(30, detection_frequency))
    save_frequency = max(1, min(60, save_frequency))

    if not model_id:
        return jsonify({"success": False, "message": "请选择模型"}), 400

    # 获取模型信息
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()

    if not model_data:
        conn.close()
        return jsonify({"success": False, "message": "模型不存在"}), 404

    # 检查模型文件是否存在
    if not os.path.exists(model_data['path']):
        conn.close()
        return jsonify({"success": False, "message": f"模型文件不存在: {model_data['path']}"}), 404

    try:
        # 创建唯一会话ID
        session_id = str(uuid.uuid4())

        # 加载YOLO模型
        model = YOLO(model_data['path'])

        # 初始化录制视频（如果需要）
        record_path = None
        video_writer = None

        if record_video:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            record_filename = f"realtime_{current_user['id']}_{timestamp}.mp4"
            record_path = os.path.join(app.config['REALTIME_RESULTS_FOLDER'], record_filename)

            # 初始化视频写入器（宽度和高度将在第一帧时设置）
            # video_writer会在第一帧到达时初始化

        # 创建会话数据
        session_data = {
            'user_id': current_user['id'],
            'model_id': model_id,
            'model': model,
            'start_time': time.time(),
            'frame_count': 0,
            'detection_count': 0,
            'categories': set(),
            'confidence_sum': 0,
            'status': 'running',
            'last_result': None,
            'record_path': record_path,
            'video_writer': video_writer,
            'video_initialized': False,
            'remark': remark,
            'detection_threshold': detection_threshold,
            'detection_frequency': detection_frequency,
            'save_frequency': save_frequency,
            'frame_to_write': 0,  # 记录当前是否需要写入帧
            'last_frame_time': time.time()  # 记录上一帧时间以计算实际帧率
        }

        # 存储会话
        realtime_sessions[session_id] = session_data

        # 更新模型使用次数
        cursor.execute("UPDATE models SET usage_count = usage_count + 1 WHERE id = %s", (model_id,))
        conn.commit()
        conn.close()

        app.logger.info(f"实时监测会话创建成功 - 会话ID: {session_id}")

        return jsonify({
            "success": True,
            "message": "实时监测已开始",
            "sessionId": session_id,
            "modelName": model_data['name']
        })

    except Exception as e:
        app.logger.error(f"启动实时监测失败: {str(e)}", exc_info=True)
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return jsonify({"success": False, "message": f"启动失败: {str(e)}"}), 500


# 处理实时视频帧
@app.route('/api/realtime/frame', methods=['POST'])
@token_required
def process_realtime_frame(current_user):
    session_id = request.form.get('sessionId')

    if not session_id or session_id not in realtime_sessions:
        return jsonify({"success": False, "message": "监测会话不存在或已过期"}), 404

    session = realtime_sessions[session_id]

    # 检查会话是否属于当前用户
    if session['user_id'] != current_user['id']:
        return jsonify({"success": False, "message": "无权访问此会话"}), 403

    # 检查会话状态
    if session['status'] != 'running':
        return jsonify({"success": False, "message": "监测会话已停止"}), 400

    # 获取上传的图像帧
    if 'frame' not in request.files:
        return jsonify({"success": False, "message": "未提供视频帧"}), 400

    frame_file = request.files['frame']

    try:
        # 读取图像数据
        frame_bytes = frame_file.read()
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "无法解码图像帧"}), 400

        # 初始化视频写入器（如果需要且尚未初始化）
        if session['record_path'] and not session['video_initialized']:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # 使用用户设置的保存频率作为视频的帧率
            # 这将确保视频播放时长与实际监测时长一致
            # 例如：如果用户设置每10秒保存一帧，那么视频的帧率应该是1/10 = 0.1
            # 但由于大多数播放器不支持太低的帧率，我们设置最低为1 FPS
            video_fps = max(1.0, 1.0 / session['save_frequency'])
            app.logger.info(f"实时监测视频帧率设置为: {video_fps} FPS (保存频率: {session['save_frequency']}秒/帧)")

            session['video_writer'] = cv2.VideoWriter(session['record_path'], fourcc, video_fps, (width, height))
            session['video_initialized'] = True
            session['frame_to_write'] = 0

        # 计算从上一帧到当前帧的时间间隔
        current_time = time.time()
        elapsed_since_last_frame = current_time - session.get('last_frame_time', current_time)
        session['last_frame_time'] = current_time

        # 根据保存频率决定是否写入当前帧
        # 累计时间，只有当累计时间达到或超过保存频率时才写入帧
        session['frame_to_write'] += elapsed_since_last_frame
        should_write_frame = session['frame_to_write'] >= session['save_frequency']

        if should_write_frame:
            # 重置累计时间，减去已用的保存周期
            session['frame_to_write'] -= session['save_frequency']

        # 帧计数增加
        session['frame_count'] += 1

        # 将图片转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用YOLO模型进行检测
        model = session['model']
        results = model.predict(frame_rgb, conf=0.25)[0]

        boxes = results.boxes.xyxy  # 获取边界框坐标
        cls_ids = results.boxes.cls  # 获取类别ID
        confidences = results.boxes.conf  # 获取置信度

        # 存储检测结果
        detections = []
        colors = {}

        # 处理检测结果
        for box, cls_id, conf in zip(boxes, cls_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            cls_name = model.names[int(cls_id)]
            session['categories'].add(cls_name)
            session['confidence_sum'] += conf.item()
            session['detection_count'] += 1

            # 为新类别分配颜色
            if cls_name not in colors:
                colors[cls_name] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls_name], 2)

            # 添加标签
            label = f"{cls_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # 绘制标签背景
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                colors[cls_name],
                -1
            )

            # 绘制标签文本
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # 添加到检测列表
            detections.append({
                'box': [x1, y1, x2, y2],
                'class': cls_name,
                'confidence': float(conf)
            })

        # 如果录制视频，写入当前帧
        if session['video_writer'] and should_write_frame:
            session['video_writer'].write(frame)
            app.logger.debug(f"写入帧到视频，当前帧计数: {session['frame_count']}")

        # 将结果帧编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 更新会话的最后结果
        session['last_result'] = {
            'detections': detections,
            'frame_count': session['frame_count'],
            'detection_count': len(detections)
        }

        # 计算当前统计信息
        total_detections = session['detection_count']
        avg_confidence = session['confidence_sum'] / total_detections if total_detections > 0 else 0
        categories = list(session['categories'])
        elapsed_time = time.time() - session['start_time']

        # 从字节流创建响应
        response = make_response(frame_bytes)
        response.headers.set('Content-Type', 'image/jpeg')

        # 添加检测结果作为HTTP头（前端可以解析）
        response.headers.set('X-Detections-Count', str(len(detections)))
        response.headers.set('X-Total-Frames', str(session['frame_count']))
        response.headers.set('X-Total-Detections', str(total_detections))
        response.headers.set('X-Avg-Confidence', str(avg_confidence))
        response.headers.set('X-Categories', ','.join(categories))
        response.headers.set('X-Elapsed-Time', str(elapsed_time))

        return response

    except Exception as e:
        app.logger.error(f"处理实时帧失败: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"处理失败: {str(e)}"}), 500


# 停止实时监测
@app.route('/api/realtime/stop', methods=['POST'])
@token_required
def stop_realtime_monitoring(current_user):
    data = request.json
    session_id = data.get('sessionId')

    if not session_id or session_id not in realtime_sessions:
        return jsonify({"success": False, "message": "监测会话不存在或已过期"}), 404

    session = realtime_sessions[session_id]

    # 检查会话是否属于当前用户
    if session['user_id'] != current_user['id']:
        return jsonify({"success": False, "message": "无权访问此会话"}), 403

    try:
        # 标记会话为已停止
        session['status'] = 'stopped'

        # 计算统计信息
        total_frames = session['frame_count']
        total_detections = session['detection_count']
        avg_confidence = session['confidence_sum'] / total_detections if total_detections > 0 else 0
        avg_accuracy = avg_confidence * 100  # 转换为百分比
        categories = list(session['categories'])
        elapsed_time = time.time() - session['start_time']

        # 关闭视频写入器（如果有）
        if session['video_writer']:
            session['video_writer'].release()

        # 保存缩略图（如果有最后一帧）
        thumbnail_path = None
        if session['last_result']:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            thumbnail_filename = f"realtime_thumb_{current_user['id']}_{timestamp}.jpg"
            thumbnail_path = os.path.join(app.config['REALTIME_RESULTS_FOLDER'], thumbnail_filename)

            # 从最后处理的帧创建缩略图 (假设这里有最后一帧的处理过的图像)
            # 这里实际实现时需要保存最后一帧的处理过图像

        # 记录到数据库
        conn = get_db_connection()
        cursor = conn.cursor()

        # 创建一个新的图像记录用于关联
        cursor.execute(
            """INSERT INTO images 
               (user_id, original_filename, storage_filename, upload_path, created_at, file_type) 
               VALUES (%s, %s, %s, %s, NOW(), %s)""",
            (current_user['id'], "实时监测录像",
             os.path.basename(session['record_path']) if session['record_path'] else "",
             session['record_path'] if session['record_path'] else "", 'realtime')
        )
        image_id = cursor.lastrowid

        # 记录处理历史
        result_file_size = os.path.getsize(session['record_path']) if session['record_path'] and os.path.exists(
            session['record_path']) else 0
        model_id = session.get('model_id', None)
        if not model_id:
            # 尝试从模型对象获取id（如果有）
            model_id = getattr(session.get('model'), 'id', None)
        if not model_id:
            # 兜底：查找images表中最新的realtime类型图片的model_id（可选）
            model_id = 0  # 或者你可以选择一个默认模型id
        cursor.execute(
            """INSERT INTO processing_history 
               (user_id, image_id, model_id, result_path, accuracy, processing_time, created_at, 
                file_type, status, progress, categories, object_count, data_size) 
               VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)""",
            (current_user['id'], image_id, model_id, session['record_path'], avg_accuracy, elapsed_time, 'realtime',
             'completed', 100, json.dumps(categories), total_detections, result_file_size)
        )

        history_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # 清理资源（从内存中删除会话）
        del realtime_sessions[session_id]

        return jsonify({
            "success": True,
            "message": "实时监测已停止",
            "result": {
                "id": history_id,
                "totalFrames": total_frames,
                "totalDetections": total_detections,
                "avgAccuracy": avg_accuracy,
                "processingTime": elapsed_time,
                "categories": categories,
                "resultPath": session['record_path'] if session['record_path'] else thumbnail_path
            }
        })

    except Exception as e:
        app.logger.error(f"停止实时监测失败: {str(e)}", exc_info=True)
        # 如果出错，还是需要清理资源
        try:
            if session.get('video_writer'):
                session['video_writer'].release()
            del realtime_sessions[session_id]
        except:
            pass

        return jsonify({"success": False, "message": f"停止失败: {str(e)}"}), 500


@app.route('/api/realtime/view/<int:history_id>', methods=['GET'])
def view_realtime_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')

    if not token:
        return jsonify({"success": False, "message": "缺少Token"}), 401

    try:
        # 验证token
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user_id = payload['sub']

        # 获取处理结果视频路径
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT result_path, status 
               FROM processing_history 
               WHERE id = %s AND user_id = %s AND file_type = 'realtime'""",
            (history_id, user_id)
        )
        record = cursor.fetchone()
        conn.close()

        if not record:
            return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

        # 检查处理状态
        if record['status'] != 'completed':
            return jsonify({"success": False, "message": "实时监测视频处理尚未完成", "status": record['status']}), 400

        result_path = record['result_path']

        # 检查结果文件是否存在
        if not os.path.exists(result_path):
            app.logger.error(f"实时监测视频结果文件不存在: {result_path}")
            return jsonify({"success": False, "message": "找不到实时监测视频结果文件"}), 404

        # 检查文件大小
        if os.path.getsize(result_path) == 0:
            app.logger.error(f"实时监测视频结果文件大小为零: {result_path}")
            return jsonify({"success": False, "message": "实时监测视频结果文件无效"}), 400

        # 设置正确的缓存控制头和范围请求支持
        response = send_file(
            result_path,
            mimetype='video/mp4',
            conditional=True  # 启用If-Range/Range请求支持
        )

        # 添加必要的头信息
        response.headers['Accept-Ranges'] = 'bytes'  # 支持范围请求
        response.headers['Access-Control-Allow-Origin'] = '*'  # CORS支持

        # 设置缓存控制
        response.headers['Cache-Control'] = 'public, max-age=3600'  # 缓存一小时
        app.logger.info(f"返回实时监测视频预览文件: {result_path}")

        return response

    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "message": "Token已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "message": "Token无效"}), 401
    except Exception as e:
        app.logger.error(f"查看实时监测结果出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"获取实时监测视频失败: {str(e)}"}), 500


@app.route('/api/realtime/download/<int:history_id>', methods=['GET'])
def download_realtime_result(history_id):
    # 从URL参数中获取token
    token = request.args.get('token')
    if not token:
        return jsonify({"error": "未提供认证令牌"}), 401

    try:
        # 验证token
        data = jwt.decode(token, app.config['JWT_SECRET'], algorithms=["HS256"])
        user_id = data['sub']  # 使用'sub'字段而不是'user_id'

        conn = get_db_connection()
        # 查询历史记录
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM processing_history WHERE id = %s AND user_id = %s AND file_type = 'realtime'",
            (history_id, user_id)
        )
        history = cursor.fetchone()
        conn.close()

        if not history:
            return jsonify({"error": "未找到对应的实时监测记录或无权访问"}), 404

        # 获取结果文件路径
        result_path = history['result_path']
        if not result_path or not os.path.exists(result_path):
            return jsonify({"error": "结果视频文件不存在"}), 404

        # 设置下载文件名 - 修复字段名称问题
        # 尝试不同可能的字段名称
        if 'original_filename' in history and history['original_filename']:
            original_filename = history['original_filename']
        elif 'filename' in history and history['filename']:
            original_filename = history['filename']
        else:
            # 如果找不到文件名，使用默认名称加上历史记录ID
            original_filename = f"实时监测视频_{history_id}.mp4"

        download_name = f"实时监测_{original_filename}"

        # 返回文件下载响应
        return send_file(result_path,
                         as_attachment=True,
                         download_name=download_name,
                         mimetype='video/mp4')

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "登录已过期，请重新登录"}), 401
    except (jwt.InvalidTokenError, Exception) as e:
        app.logger.error(f"下载实时监测结果出错: {str(e)}")
        return jsonify({"error": "处理请求时出错"}), 500


# 获取单个历史记录详情
@app.route('/api/history/<int:history_id>', methods=['GET'])
@token_required
def get_history_detail(current_user, history_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询单条记录
    cursor.execute("""
        SELECT h.*, i.original_filename, i.upload_path, i.storage_filename, m.name as model_name, m.type as model_type
        FROM processing_history h
        JOIN images i ON h.image_id = i.id
        JOIN models m ON h.model_id = m.id
        WHERE h.id = %s AND h.user_id = %s
    """, (history_id, current_user['id']))

    record = cursor.fetchone()
    conn.close()

    if not record:
        return jsonify({"success": False, "message": "记录不存在或无权访问"}), 404

    # 转换日期时间为字符串
    if 'created_at' in record and record['created_at']:
        record['created_at'] = record['created_at'].isoformat()

    return jsonify({
        "success": True,
        "record": record
    })


@app.route('/api/models/download', methods=['GET'])
@token_required
def download_model_file(current_user):
    rel_or_abs_path = request.args.get('path')
    if not rel_or_abs_path:
        return jsonify({'success': False, 'message': '缺少模型文件路径'}), 400
    # 转为相对路径
    model_root = os.path.abspath(os.path.join(app.config['MODEL_FOLDER'], '..'))  # models 目录
    abs_path = os.path.abspath(rel_or_abs_path)
    if abs_path.startswith(model_root):
        rel_path = os.path.relpath(abs_path, model_root)
    else:
        rel_path = rel_or_abs_path  # 可能本来就是相对路径
    # 拼接绝对路径
    model_path = os.path.abspath(os.path.join(model_root, rel_path))
    if not model_path.startswith(model_root):
        return jsonify({'success': False, 'message': '非法的模型文件路径'}), 403
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'message': '模型文件不存在'}), 404
    filename = os.path.basename(model_path)
    return send_file(model_path, as_attachment=True, download_name=filename)


# 管理员API：获取待处理权限申请数量
@app.route('/api/admin/permissions/count', methods=['GET'])
@token_required
def admin_get_pending_permissions_count(current_user):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询待处理的权限申请数量
    cursor.execute(
        """SELECT COUNT(*) as pending_count 
           FROM permission_requests 
           WHERE status = 'pending'"""
    )
    result = cursor.fetchone()
    pending_count = result['pending_count'] if result else 0
    conn.close()

    return jsonify({
        "success": True,
        "pendingCount": pending_count
    })


# 管理员API：获取模型公开申请列表
@app.route('/api/admin/model-publish-requests', methods=['GET'])
@token_required
def admin_get_model_publish_requests(current_user):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    # 获取分页参数
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('perPage', 10))
    offset = (page - 1) * per_page

    # 获取状态过滤参数
    status = request.args.get('status', 'pending')  # pending, approved, rejected, all

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 构建查询条件
        status_condition = ""
        if status != 'all':
            status_condition = f"AND mpr.status = '{status}'"

        # 查询总数
        cursor.execute(f"""
            SELECT COUNT(*) as total
            FROM model_publish_requests mpr
            WHERE 1=1 {status_condition}
        """)
        total = cursor.fetchone()['total']

        # 获取申请列表
        cursor.execute(f"""
            SELECT 
                mpr.id, 
                mpr.model_id, 
                mpr.user_id, 
                mpr.reason, 
                mpr.status, 
                mpr.admin_comment, 
                mpr.created_at,
                mpr.updated_at,
                m.name as model_name,
                m.type as model_type,
                u.username as username
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            JOIN users u ON mpr.user_id = u.id
            WHERE 1=1 {status_condition}
            ORDER BY 
                CASE WHEN mpr.status = 'pending' THEN 0 ELSE 1 END,
                mpr.created_at DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))

        requests = []
        for row in cursor.fetchall():
            requests.append({
                "id": row['id'],
                "model_id": row['model_id'],
                "user_id": row['user_id'],
                "username": row['username'],
                "model_name": row['model_name'],
                "model_type": row['model_type'],
                "reason": row['reason'],
                "status": row['status'],
                "admin_comment": row['admin_comment'],
                "created_at": row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else None,
                "updated_at": row['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if row['updated_at'] else None,
            })

        conn.close()
        return jsonify({
            "success": True,
            "requests": requests,
            "pagination": {
                "total": total,
                "page": page,
                "perPage": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        })
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"获取模型公开申请失败: {str(e)}"}), 500


# 管理员API：获取单个模型公开申请详情
@app.route('/api/admin/model-publish-requests/<int:request_id>', methods=['GET'])
@token_required
def admin_get_model_publish_request_detail(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 获取申请详情
        cursor.execute("""
            SELECT 
                mpr.id, 
                mpr.model_id, 
                mpr.user_id, 
                mpr.reason, 
                mpr.status, 
                mpr.admin_comment, 
                mpr.created_at,
                mpr.updated_at,
                m.name as model_name,
                m.type as model_type,
                m.description as model_description,
                m.example_image as model_example_image,
                m.result_example_image as model_result_example_image,
                m.parameters as model_parameters,
                m.instructions as model_instructions,
                m.path as model_path,
                u.username as username,
                u.email as user_email
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            JOIN users u ON mpr.user_id = u.id
            WHERE mpr.id = %s
        """, (request_id,))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到指定的申请记录"}), 404

        # 格式化响应数据
        result = {
            "id": request_data['id'],
            "model_id": request_data['model_id'],
            "user_id": request_data['user_id'],
            "username": request_data['username'],
            "user_email": request_data['user_email'],
            "model_name": request_data['model_name'],
            "model_type": request_data['model_type'],
            "model_description": request_data['model_description'],
            "model_example_image": request_data['model_example_image'],
            "model_result_example_image": request_data['model_result_example_image'],
            "model_parameters": request_data['model_parameters'],
            "model_instructions": request_data['model_instructions'],
            "model_path": request_data['model_path'],
            "reason": request_data['reason'],
            "status": request_data['status'],
            "admin_comment": request_data['admin_comment'],
            "created_at": request_data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if request_data[
                'created_at'] else None,
            "updated_at": request_data['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if request_data[
                'updated_at'] else None,
        }

        conn.close()
        return jsonify({
            "success": True,
            "request": result
        })
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"获取模型公开申请详情失败: {str(e)}"}), 500


# 管理员API：批准模型公开申请
@app.route('/api/admin/model-publish-requests/<int:request_id>/approve', methods=['POST'])
@token_required
def admin_approve_model_publish_request(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    # 获取管理员评论
    data = request.json
    admin_comment = data.get('adminComment', '')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查申请是否存在且状态为pending
        cursor.execute("""
            SELECT mpr.id, mpr.model_id, mpr.user_id, m.name as model_name, u.username
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            JOIN users u ON mpr.user_id = u.id
            WHERE mpr.id = %s AND mpr.status = 'pending'
        """, (request_id,))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到待处理的申请或该申请已被处理"}), 404

        model_id = request_data['model_id']
        user_id = request_data['user_id']
        model_name = request_data['model_name']
        username = request_data['username']

        # 更新申请状态
        cursor.execute("""
            UPDATE model_publish_requests 
            SET status = 'approved', admin_comment = %s, updated_at = NOW() 
            WHERE id = %s
        """, (admin_comment, request_id))

        # 更新模型状态为公开
        cursor.execute("""
            UPDATE models
            SET is_shared = 1
            WHERE id = %s
        """, (model_id,))

        # 创建通知给用户
        cursor.execute("""
            INSERT INTO notifications 
            (user_id, title, content, type, is_read, created_at) 
            VALUES (%s, '模型公开申请已批准', %s, 'model_publish', 0, NOW())
        """, (user_id, f"您的模型 \"{model_name}\" 公开申请已被批准，现在其他用户可以使用该模型"))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型公开申请已批准"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"处理模型公开申请失败: {str(e)}"}), 500


# 管理员API：拒绝模型公开申请
@app.route('/api/admin/model-publish-requests/<int:request_id>/reject', methods=['POST'])
@token_required
def admin_reject_model_publish_request(current_user, request_id):
    # 验证是否是管理员
    if not current_user.get('is_admin'):
        return jsonify({"success": False, "message": "无权访问，需要管理员权限"}), 403

    # 获取管理员评论
    data = request.json
    admin_comment = data.get('adminComment', '')

    if not admin_comment.strip():
        return jsonify({"success": False, "message": "拒绝理由不能为空"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查申请是否存在且状态为pending
        cursor.execute("""
            SELECT mpr.id, mpr.model_id, mpr.user_id, m.name as model_name, u.username
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            JOIN users u ON mpr.user_id = u.id
            WHERE mpr.id = %s AND mpr.status = 'pending'
        """, (request_id,))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到待处理的申请或该申请已被处理"}), 404

        model_id = request_data['model_id']
        user_id = request_data['user_id']
        model_name = request_data['model_name']
        username = request_data['username']

        # 更新申请状态
        cursor.execute("""
            UPDATE model_publish_requests 
            SET status = 'rejected', admin_comment = %s, updated_at = NOW() 
            WHERE id = %s
        """, (admin_comment, request_id))

        # 模型保持私有状态，不需要更新

        # 创建通知给用户
        cursor.execute("""
            INSERT INTO notifications 
            (user_id, title, content, type, is_read, created_at) 
            VALUES (%s, '模型公开申请已被拒绝', %s, 'model_publish', 0, NOW())
        """, (user_id, f"您的模型 \"{model_name}\" 公开申请已被拒绝，原因：{admin_comment}"))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型公开申请已拒绝"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"处理模型公开申请失败: {str(e)}"}), 500


# 用户API：获取自己的模型公开申请记录
@app.route('/api/model-publish-requests', methods=['GET'])
@token_required
def get_user_model_publish_requests(current_user):
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('perPage', 10, type=int)
    offset = (page - 1) * per_page

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 获取总记录数
        cursor.execute("""
            SELECT COUNT(*) as total
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            WHERE mpr.user_id = %s
        """, (current_user['id'],))

        total = cursor.fetchone()['total']

        # 分页获取申请记录
        cursor.execute("""
            SELECT 
                mpr.id, mpr.model_id, mpr.reason, mpr.status, mpr.admin_comment,
                mpr.created_at, mpr.updated_at,
                m.name as model_name, m.type as model_type
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            WHERE mpr.user_id = %s
            ORDER BY mpr.created_at DESC
            LIMIT %s OFFSET %s
        """, (current_user['id'], per_page, offset))

        requests = []
        for row in cursor.fetchall():
            requests.append({
                "id": row['id'],
                "model_id": row['model_id'],
                "model_name": row['model_name'],
                "model_type": row['model_type'],
                "reason": row['reason'],
                "status": row['status'],
                "admin_comment": row['admin_comment'],
                "created_at": row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else None,
                "updated_at": row['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if row['updated_at'] else None,
            })

        conn.close()
        return jsonify({
            "success": True,
            "requests": requests,
            "pagination": {
                "total": total,
                "page": page,
                "perPage": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        })
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"获取模型公开申请列表失败: {str(e)}"}), 500


# 获取模型公开申请详情（用户版本）
@app.route('/api/model-publish-requests/<int:request_id>', methods=['GET'])
@token_required
def get_model_publish_request_detail(current_user, request_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 获取申请详情，并确保只返回用户自己的申请
        cursor.execute("""
            SELECT 
                mpr.*, 
                m.name as model_name, 
                m.type as model_type, 
                m.description as model_description,
                m.example_image as model_example_image,
                m.result_example_image as model_result_example_image,
                m.parameters as model_parameters,
                m.instructions as model_instructions,
                m.path as model_path
            FROM model_publish_requests mpr
            JOIN models m ON mpr.model_id = m.id
            WHERE mpr.id = %s AND mpr.user_id = %s
        """, (request_id, current_user['id']))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到申请记录或无权限查看"}), 404

        # 格式化响应数据
        result = {
            "id": request_data['id'],
            "model_id": request_data['model_id'],
            "model_name": request_data['model_name'],
            "model_type": request_data['model_type'],
            "model_description": request_data['model_description'],
            "model_example_image": request_data['model_example_image'],
            "model_result_example_image": request_data['model_result_example_image'],
            "model_parameters": request_data['model_parameters'],
            "model_instructions": request_data['model_instructions'],
            "model_path": request_data['model_path'],
            "reason": request_data['reason'],
            "status": request_data['status'],
            "admin_comment": request_data['admin_comment'],
            "created_at": request_data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if request_data[
                'created_at'] else None,
            "updated_at": request_data['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if request_data[
                'updated_at'] else None,
        }

        conn.close()
        return jsonify({
            "success": True,
            "request": result
        })
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"获取模型公开申请详情失败: {str(e)}"}), 500


# 修改模型公开申请
@app.route('/api/model-publish-requests/<int:request_id>', methods=['PUT'])
@token_required
def update_model_publish_request(current_user, request_id):
    # 获取修改后的理由
    data = request.json
    new_reason = data.get('reason', '')

    if not new_reason.strip():
        return jsonify({"success": False, "message": "申请理由不能为空"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查申请是否存在、属于当前用户且状态为pending
        cursor.execute("""
            SELECT id
            FROM model_publish_requests
            WHERE id = %s AND user_id = %s AND status = 'pending'
        """, (request_id, current_user['id']))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到待处理的申请，或该申请不属于您，或已被处理"}), 404

        # 更新申请理由
        cursor.execute("""
            UPDATE model_publish_requests 
            SET reason = %s, updated_at = NOW() 
            WHERE id = %s
        """, (new_reason, request_id))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型公开申请已更新"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"更新模型公开申请失败: {str(e)}"}), 500


# 撤销模型公开申请
@app.route('/api/model-publish-requests/<int:request_id>', methods=['DELETE'])
@token_required
def cancel_model_publish_request(current_user, request_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查申请是否存在、属于当前用户且状态为pending
        cursor.execute("""
            SELECT id
            FROM model_publish_requests
            WHERE id = %s AND user_id = %s AND status = 'pending'
        """, (request_id, current_user['id']))

        request_data = cursor.fetchone()
        if not request_data:
            conn.close()
            return jsonify({"success": False, "message": "未找到待处理的申请，或该申请不属于您，或已被处理"}), 404

        # 删除申请记录
        cursor.execute("""
            DELETE FROM model_publish_requests 
            WHERE id = %s
        """, (request_id,))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "模型公开申请已撤销"
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": f"撤销模型公开申请失败: {str(e)}"}), 500


# 主程序入口
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='遥感图像识别系统后端服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机名')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    print(f"启动遥感图像识别系统后端服务，主机名：{args.host}，端口：{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)