from django.db import models


class UserModel(models.Model):
    username = models.CharField(max_length=150, unique=True)  # 延长用户名长度限制
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # 适应哈希密码长度

    # 添加创建时间字段用于监控和分析
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 添加索引以提高查询性能
        indexes = [
            models.Index(fields=['username']),
            models.Index(fields=['email']),
        ]
        # 明确指定表名
        db_table = 'user_accounts'