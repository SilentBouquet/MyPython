import jwt
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDc2NjIxMjUsImlhdCI6MTc0NzA1NzMyNSwic3ViIjo0LCJ1c2VybmFtZSI6Ilx1ODg4MVx1ODk3ZiJ9.hQoc2_sQXeONhnnOMOtrQ__Z8j1qOOopHbeJa8W96TA"
secret = "your_jwt_secret_key"  # 换成你实际用的密钥
payload = jwt.decode(token, secret, algorithms=['HS256'])
print(payload)