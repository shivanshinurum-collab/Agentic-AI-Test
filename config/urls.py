from django.contrib import admin
from django.urls import path, include
from ai_service.views import ChatView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ai_service.urls')),
    path('chat/', ChatView.as_view(), name='chat-page'),
    path('', ChatView.as_view(), name='home-chat'),
]

