from django.urls import path
from .views import (
    SystemStatusView,
    ComputeBenchmarkView,
    PredictView,
    ModelStatusView,
    ModelLoadView,
    AgentRunView,
    AgentToolsView,
    AgentMemoryView,
    AgentDashboardView,
)

urlpatterns = [
    # System & Model endpoints
    path('status/', SystemStatusView.as_view(), name='system-status'),
    path('benchmark/', ComputeBenchmarkView.as_view(), name='compute-benchmark'),
    path('predict/', PredictView.as_view(), name='ai-predict'),

    # GGUF Model Management endpoints
    path('model/status/', ModelStatusView.as_view(), name='model-status'),
    path('model/load/', ModelLoadView.as_view(), name='model-load'),

    # Agentic AI endpoints
    path('agent/run/', AgentRunView.as_view(), name='agent-run'),
    path('agent/tools/', AgentToolsView.as_view(), name='agent-tools'),
    path('agent/memory/', AgentMemoryView.as_view(), name='agent-memory'),
    path('agent/dashboard/', AgentDashboardView.as_view(), name='agent-dashboard'),
    path('agent/', AgentDashboardView.as_view(), name='agent-dashboard-root'),
]
