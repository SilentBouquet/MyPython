import json
import logging
from .model import UserModel
from django.http import JsonResponse
from django.views import View
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError, transaction
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class User(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'code': 400, 'message': "无效的JSON数据"})

        name = data.get("name", "").strip()
        password = data.get("password", "")
        email = data.get("email", "").strip()

        if not name or not password or not email:
            return JsonResponse({
                'code': 400,
                'message': "缺少必要参数"
            })

        # 验证邮箱格式
        try:
            validate_email(email)
        except ValidationError:
            return JsonResponse({'code': 400, 'message': "邮箱格式不正确"})

        # 验证密码强度
        if len(password) < 8:
            return JsonResponse({'code': 400, 'message': "密码长度至少8位"})

        # 使用原子事务处理注册
        try:
            with transaction.atomic():
                user_obj = UserModel.objects.create(
                    username=name,
                    password=make_password(password),
                    email=email
                )
                return JsonResponse({
                    'code': 200,
                    "message": "用户注册成功"
                })
        except IntegrityError as e:
            # 处理唯一约束违反
            if 'username' in str(e):
                return JsonResponse({
                    'code': 400,
                    'message': "用户名已存在"
                })
            elif 'email' in str(e):
                return JsonResponse({
                    'code': 400,
                    'message': "邮箱已存在"
                })
            else:
                logger.error(f"注册完整性错误: {str(e)}")
                return JsonResponse({
                    'code': 400,
                    'message': "注册失败，请稍后重试"
                })
        except Exception as e:
            logger.error(f"用户注册异常: {str(e)}")
            return JsonResponse({
                'code': 500,
                'message': "系统错误，请稍后重试"
            })