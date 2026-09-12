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
        Deep NLP & Sentiment Analysis powered by GGUF Local Model Engine (or enhanced rule-assisted fallback).
        """
        text_clean = text.strip()
        if not text_clean:
            return {"error": "Text cannot be empty"}
        
        # Try local GGUF model inference first
        try:
            from ai_service.gguf_engine import gguf_engine
            if gguf_engine.is_model_available():
                system_prompt = (
                    "You are an expert NLP Sentiment & Emotion Analyzer. "
                    "Analyze the given text and respond strictly in valid JSON format with keys:\n"
                    '{"sentiment": "POSITIVE"|"NEGATIVE"|"NEUTRAL"|"MIXED", "confidence": float_0_to_1, "emotions": [string], "summary_explanation": string}'
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze sentiment and emotion for: '{text_clean}'"}
                ]
                res = gguf_engine.chat_completion(messages=messages, temperature=0.1, max_tokens=300)
                if res.get("success") and res.get("content"):
                    import json
                    raw_content = res.get("content").strip()
                    # Extract JSON block
                    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        return {
                            "input_text": text,
                            "sentiment": parsed.get("sentiment", "NEUTRAL"),
                            "confidence": round(float(parsed.get("confidence", 0.85)), 4),
                            "emotions": parsed.get("emotions", []),
                            "analysis": parsed.get("summary_explanation", "AI analyzed text sentiment."),
                            "word_count": len(text_clean.split()),
                            "engine": "GGUF Qwen Model (Apple Silicon GPU)",
                            "device": self.device_name,
                        }
        except Exception:
            pass

        # Fallback sentiment scoring
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
            confidence = min(0.65 + score * 0.35, 0.99)
        elif score < -0.05:
            sentiment = "NEGATIVE"
            confidence = min(0.65 + abs(score) * 0.35, 0.99)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.55
            
        return {
            "input_text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "word_count": len(words),
            "engine": "Standard Rule Analyzer",
            "device": self.device_name,
        }

import re

# Global singleton accessor
ai_service = AIModelService.get_instance()
