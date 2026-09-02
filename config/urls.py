"""
URL configuration for config project.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ai_service.urls')),
    path('', RedirectView.as_view(url='/api/agent/', permanent=False), name='index-redirect'),
]
