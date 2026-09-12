import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"

class GGUFEngine:
    """
    High-performance GGUF Model Runtime Engine with Apple Silicon Metal (GPU) acceleration.
    Supports Qwen3-4B-Q4_K_M.gguf and other quantized GGUF architectures.
    """
    _instance = None

    def __init__(self):
        self.model = None
        self.loaded_model_path: Optional[str] = None
        self.gpu_layers: int = -1  # -1 offloads all layers to Apple Silicon Metal GPU
        self.context_size: int = 8192
        self.models_dir = DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "GGUFEngine":
        if cls._instance is None:
            cls._instance = GGUFEngine()
        return cls._instance

    def find_available_models(self) -> List[Dict[str, Any]]:
        """
        Scans models directory for all .gguf model files.
        """
        gguf_files = list(self.models_dir.glob("*.gguf"))
        result = []
        for p in gguf_files:
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            result.append({
                "filename": p.name,
                "path": str(p.resolve()),
                "size_mb": size_mb,
                "is_qwen": "qwen" in p.name.lower(),
                "is_active": str(p.resolve()) == self.loaded_model_path
            })
        return result

    def get_default_model_path(self) -> Optional[str]:
        """
        Returns the path to Qwen3-4B-Q4_K_M.gguf or first available .gguf file.
        """
        preferred_names = [
            "Qwen3-4B-Q4_K_M.gguf",
            "qwen3-4b-q4_k_m.gguf",
            "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        ]
        for name in preferred_names:
            p = self.models_dir / name
            if p.exists():
                return str(p.resolve())

        # Fallback to any .gguf file found
        any_models = list(self.models_dir.glob("*.gguf"))
        if any_models:
            return str(any_models[0].resolve())

        return None

    def is_model_available(self, custom_path: Optional[str] = None) -> bool:
        target = custom_path or self.get_default_model_path()
        return bool(target and Path(target).exists())

    def get_status(self) -> Dict[str, Any]:
        """
        Returns runtime status of GGUF model and Apple Silicon GPU state.
        """
        llama_cpp_installed = False
        try:
            import llama_cpp
            llama_cpp_installed = True
        except ImportError:
            pass

        available_models = self.find_available_models()
        default_path = self.get_default_model_path()

        return {
            "llama_cpp_installed": llama_cpp_installed,
            "is_loaded": self.model is not None,
            "loaded_model_path": self.loaded_model_path,
            "default_model_path": default_path,
            "target_model_name": "Qwen3-4B-Q4_K_M.gguf",
            "target_model_found": self.is_model_available(),
            "models_directory": str(self.models_dir),
            "available_models": available_models,
            "gpu_acceleration": "Apple Silicon Metal (MPS/GPU)",
            "gpu_layers": self.gpu_layers,
            "context_size": self.context_size
        }

    def load_model(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_threads: int = 8
    ) -> Dict[str, Any]:
        """
        Loads GGUF model into Apple Silicon Metal GPU unified memory.
        """
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError:
            return {
                "success": False,
                "error": "llama-cpp-python is not yet installed. Run `pip install llama-cpp-python`."
            }

        target_path = model_path or self.get_default_model_path()
        if not target_path or not os.path.exists(target_path):
            expected = model_path or str(self.models_dir / "Qwen3-4B-Q4_K_M.gguf")
            return {
                "success": False,
                "error": f"Model file not found at: '{expected}'. Please place your Qwen3-4B-Q4_K_M.gguf in the models/ directory."
            }

        # If already loaded same model with same settings, reuse
        if self.model is not None and self.loaded_model_path == target_path:
            return {
                "success": True,
                "message": "Model already loaded and active in Metal GPU memory.",
                "model_path": target_path
            }

        try:
            print(f"[GGUF Engine] Loading {target_path} into Metal GPU (n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})...")
            self.model = Llama(
                model_path=target_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
            self.loaded_model_path = target_path
            self.gpu_layers = n_gpu_layers
            self.context_size = n_ctx
            print(f"[GGUF Engine] Successfully loaded model: {os.path.basename(target_path)}")
            return {
                "success": True,
                "message": f"Successfully loaded {os.path.basename(target_path)} on Metal GPU.",
                "model_path": target_path,
                "gpu_layers": n_gpu_layers,
                "context_size": n_ctx
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize GGUF model on GPU: {str(e)}"
            }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 3500,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes chat completion on the loaded Qwen GGUF model.
        """
        if self.model is None:
            load_res = self.load_model()
            if not load_res.get("success"):
                return load_res

        stop_sequences = stop or ["<|im_end|>", "<|endoftext|>", "\nObservation:"]
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            content = response["choices"][0]["message"]["content"]
            return {
                "success": True,
                "content": content,
                "usage": response.get("usage", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference execution error: {str(e)}"
            }

# Global singleton
gguf_engine = GGUFEngine.get_instance()
