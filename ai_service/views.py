import sys
import platform
import torch
import django
from django.shortcuts import render
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .model_manager import ai_service
from .gguf_engine import gguf_engine
from .agent import agent_executor, default_registry, MemoryManager

class SystemStatusView(APIView):
    """
    Returns the backend system health and AI runtime details.
    """
    def get(self, request):
        gguf_status = gguf_engine.get_status()
        return Response({
            "status": "healthy",
            "python_version": sys.version,
            "django_version": django.get_version(),
            "pytorch_version": torch.__version__,
            "accelerator": {
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "active_device": ai_service.device_name,
            },
            "agentic_ai": {
                "enabled": True,
                "tools_registered": len(default_registry.list_tools()),
                "gguf_model": gguf_status
            },
            "platform": platform.platform(),
        }, status=status.HTTP_200_OK)


class ComputeBenchmarkView(APIView):
    """
    Executes a live PyTorch tensor operation on Apple Silicon MPS GPU.
    """
    def post(self, request):
        matrix_size = request.data.get("matrix_size", 1000)
        try:
            matrix_size = int(matrix_size)
            if matrix_size > 5000:
                return Response(
                    {"error": "matrix_size must be <= 5000"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            result = ai_service.run_tensor_computation(matrix_size)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PredictView(APIView):
    """
    Receives text payload and runs AI model inference.
    """
    def post(self, request):
        text = request.data.get("text", "")
        if not text:
            return Response(
                {"error": "Please provide a 'text' field in the JSON body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = ai_service.predict_sentiment_simple(text)
        return Response(result, status=status.HTTP_200_OK)


# ==========================================
# GGUF Model Management Views
# ==========================================

class ModelStatusView(APIView):
    """
    Returns the status of local GGUF models and GPU offloading state.
    """
    def get(self, request):
        status_data = gguf_engine.get_status()
        return Response(status_data, status=status.HTTP_200_OK)


class ModelLoadView(APIView):
    """
    Loads or reloads a specific GGUF model into Apple Silicon Metal GPU memory.
    """
    def post(self, request):
        model_path = request.data.get("model_path")
        n_gpu_layers = request.data.get("n_gpu_layers", -1)
        n_ctx = request.data.get("n_ctx", 4096)

        try:
            n_gpu_layers = int(n_gpu_layers)
            n_ctx = int(n_ctx)
        except (ValueError, TypeError):
            n_gpu_layers = -1
            n_ctx = 4096

        load_result = gguf_engine.load_model(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )
        http_status = status.HTTP_200_OK if load_result.get("success") else status.HTTP_400_BAD_REQUEST
        return Response(load_result, status=http_status)


# ==========================================
# Agentic AI Views
# ==========================================

class AgentRunView(APIView):
    """
    Executes an autonomous Agentic AI task using ReAct planning and dynamic tool calling.
    Supports Qwen3-4B-Q4_K_M.gguf with Apple Silicon GPU acceleration.
    """
    def post(self, request):
        prompt = request.data.get("prompt", "")
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return Response(
                {"error": "Please provide a non-empty 'prompt' string."},
                status=status.HTTP_400_BAD_REQUEST
            )

        max_iterations = request.data.get("max_iterations", 6)
        session_id = request.data.get("session_id", "default")
        planner_mode = request.data.get("planner_mode", "gguf")
        model_path = request.data.get("model_path", None)
        llm_config = request.data.get("llm_config", None)

        try:
            max_iterations = int(max_iterations)
            if max_iterations < 1 or max_iterations > 20:
                max_iterations = 6
        except (ValueError, TypeError):
            max_iterations = 6

        try:
            result = agent_executor.run(
                prompt=prompt.strip(),
                session_id=session_id,
                max_iterations=max_iterations,
                planner_mode=planner_mode,
                model_path=model_path,
                llm_config=llm_config
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Agent execution error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AgentToolsView(APIView):
    """
    Returns the catalog of all available Agentic tools and schemas.
    """
    def get(self, request):
        tools = default_registry.list_tools()
        return Response({
            "count": len(tools),
            "tools": tools
        }, status=status.HTTP_200_OK)


class AgentMemoryView(APIView):
    """
    Manages session memory and scratchpad history.
    """
    def get(self, request):
        session_id = request.query_params.get("session_id", "default")
        memory = MemoryManager.get_session(session_id)
        return Response({
            "session_id": session_id,
            "variables": memory.get_all_vars(),
            "history": memory.get_history()
        }, status=status.HTTP_200_OK)

    def delete(self, request):
        session_id = request.data.get("session_id", "default")
        cleared = MemoryManager.clear_session(session_id)
        return Response({
            "session_id": session_id,
            "cleared": cleared
        }, status=status.HTTP_200_OK)


class AgentDashboardView(View):
    """
    Renders the visual interactive Agentic AI Dashboard.
    """
    def get(self, request):
        return render(request, "ai_service/agent_dashboard.html")
