# waterApp/urls.py

from django.urls import path
from . import views

app_name = 'waterApp'

urlpatterns = [
    path('', views.home, name='home'),
    # 다른 URL 패턴들을 여기에 추가할 수 있습니다.
    path('water_result', views.your_submission_url, name='water_result.html'),
]
