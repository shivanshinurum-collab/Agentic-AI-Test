import torch
import numpy as np

class AIModelService:
    _instance = None

    def __init__(self):
        # Auto-detect best available device: Apple Silicon (MPS), CUDA (Nvidia), or CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_name = "Apple Silicon GPU (MPS)"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = "NVIDIA CUDA GPU"
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
        
        print(f"[AI Model Service] Initialized on device: {self.device_name}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AIModelService()
        return cls._instance

    def run_tensor_computation(self, matrix_size: int = 1000) -> dict:
        """
        Runs a high-performance matrix multiplication on the active accelerator (MPS/GPU).
        """
        # Create random tensors on the accelerated device
        a = torch.randn((matrix_size, matrix_size), device=self.device)
        b = torch.randn((matrix_size, matrix_size), device=self.device)
        
        # Perform matrix multiplication
        start_event = torch.mps.current_allocated_memory() if self.device.type == "mps" else 0
        c = torch.matmul(a, b)
        
        return {
            "device": self.device_name,
            "matrix_shape": list(c.shape),
            "result_mean": float(c.mean().item()),
            "result_std": float(c.std().item()),
        }

    def predict_sentiment_simple(self, text: str) -> dict:
        """
        Sample rule-assisted embedding representation demo.
        Can be replaced with Hugging Face transformers pipeline or custom weights.
        """
        text_clean = text.strip()
        if not text_clean:
            return {"error": "Text cannot be empty"}
        
        # Expanded sentiment scoring with punctuation stripping
        pos_words = {
            "good", "great", "excellent", "awesome", "fast", "love", "loved", "lovely",
            "like", "liked", "best", "happy", "fantastic", "amazing", "wonderful", "wonderfully",
            "super", "brilliant", "helpful", "blazing", "smooth", "perfect", "perfectly", "positive"
        }
        neg_words = {
            "bad", "slow", "error", "poor", "hate", "hated", "issue", "worst", "bug",
            "terrible", "horrible", "awful", "failed", "failure", "broken", "negative"
        }
        
        raw_words = text_clean.lower().split()
        words = [w.strip(".,!?;:\"'()[]{}") for w in raw_words]
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        
        score = (pos_count - neg_count) / max(len(words), 1)
        
        if score > 0.05:
            sentiment = "POSITIVE"
            confidence = min(0.6 + score * 0.4, 0.99)
        elif score < -0.05:
            sentiment = "NEGATIVE"
            confidence = min(0.6 + abs(score) * 0.4, 0.99)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
            
        return {
            "input_text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "word_count": len(words),
            "device": self.device_name,
        }

# Global singleton accessor
ai_service = AIModelService.get_instance()
