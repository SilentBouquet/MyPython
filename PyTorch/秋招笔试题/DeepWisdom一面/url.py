from django.urls import path
from app import User

urlpatterns = [
    path('register', User.as_view()),
]