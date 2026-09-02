# AI Model & Agentic AI Backend (Django + PyTorch + Qwen3-4B GGUF + ReAct Framework)

This repository contains a high-performance Django backend integrated with PyTorch, Apple Silicon GPU acceleration (Metal / MPS), and an autonomous **Agentic AI** framework powered by **Qwen3-4B-Q4_K_M.gguf** running natively on Apple Silicon GPU via Metal offloading.

---

## 🚀 Environment Overview

Your system is configured with a **Global Python 3.12 Environment** (`~/.global-python-env`):
- `python` / `python3`: Python 3.12.14
- `pip` / `pip3`: Pip 26.2.1
- `django-admin`: Django 6.1
- `PyTorch`: 2.13.0 with Apple Silicon MPS GPU acceleration enabled
- `llama-cpp-python`: 0.3.35 with Metal GPU hardware acceleration
- `Transformers`, `Scikit-learn`, `NumPy`, `Pandas`, `Pillow`, `Django REST Framework`

To activate the environment in your shell:
```bash
source ~/.global-python-env/bin/activate
```

---

## 🦙 GGUF Local Model Integration (`Qwen3-4B-Q4_K_M.gguf`)

The backend is fully configured to execute `Qwen3-4B-Q4_K_M.gguf` with **Apple Silicon Metal GPU offloading** (`n_gpu_layers=-1`):

### How to use your Model File:
1. **Option A (Recommended)**: Drop `Qwen3-4B-Q4_K_M.gguf` inside the [`models/`](file:///Users/inurum/Documents/AI_Model/models/) folder:
   ```bash
   cp /path/to/your/Qwen3-4B-Q4_K_M.gguf /Users/inurum/Documents/AI_Model/models/
   ```
2. **Option B (Direct File Path)**: You can provide any absolute or relative path directly via the Dashboard UI or API payload:
   ```json
   {
     "prompt": "Calculate the compound interest...",
     "planner_mode": "gguf",
     "model_path": "/path/to/your/Qwen3-4B-Q4_K_M.gguf"
   }
   ```

---

## 🧠 Agentic AI Architecture

```
[User Objective] 
      │
      ▼
┌──────────────┐    Reasoning    ┌──────────────────────────────────────────────┐
│ ReAct Loop   │ ──────────────> │ Qwen3-4B-Q4_K_M.gguf (Metal GPU Accelerated) │
│ Orchestrator │ <────────────── │ (Thought + Action Plan)                      │
└──────┬───────┘                 └──────────────────────────────────────────────┘
       │
       ▼ Action (Tool Invocation)
┌──────────────────────────────────────────────────────────┐
│ Pluggable Tool Registry                                  │
│  ├─ 🗺️ Google Maps (Geocoding, Routes, Places & Distances)│
│  ├─ 🧮 Calculator (Safe AST math & statistical formulas)  │
│  ├─ 🐍 Python Interpreter (Sandboxed code execution)      │
│  ├─ 🔍 Knowledge Search (Concept & document retrieval)   │
│  ├─ 🧠 NLP Analyzer (PyTorch sentiment & token analysis) │
│  ├─ ⏰ DateTime Tool (Temporal arithmetic & timestamps)  │
│  └─ 💾 Memory Store (Session-aware context retention)    │
└──────────────────────────────────────────────────────────┘
       │ Observation
       ▼
[Self-Reflection & Final Synthesis]
```

---

## 🗺️ Google Maps Integration & High Accuracy Mode

To enable live real-time Google Maps data (live traffic, turn-by-turn directions, place ratings, and global coordinates):

### 1. Set your Google Maps API Key
You can export it in your shell or place it in your environment:
```bash
export GOOGLE_MAPS_API_KEY="your_google_maps_api_key_here"
```

Or pass it dynamically in API payloads or Django settings.

> **💡 Zero-Key Fallback Engine**: If no API key is provided, the backend seamlessly activates the built-in **High-Accuracy Spatial Engine** (using curated global landmarks, Great-Circle Haversine distance matrix, and OpenStreetMap geocoding fallback) so navigation and distance calculations always work accurately offline.

### 2. Supported Google Maps Operations:
- **`directions`**: Turn-by-turn navigation, real-time traffic durations, and distance between origins and destinations.
- **`geocode`**: Precise latitude and longitude coordinates and formatted addresses for any place or landmark.
- **`reverse_geocode`**: Convert coordinates back into street addresses and neighborhoods.
- **`places_search`**: Find top-rated cafes, restaurants, hotels, hospitals, and businesses with ratings and opening hours.
- **`distance_matrix`**: Compute multi-point distance and travel duration matrices.

---

## 🛠️ Quick Start

### 1. Run the Development Server
```bash
source ~/.global-python-env/bin/activate
python manage.py runserver 8000
```

### 2. Open the Interactive Visual Dashboard
Open your browser at:
👉 **`http://127.0.0.1:8000/`** (or `http://127.0.0.1:8000/api/agent/dashboard/`)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `http://127.0.0.1:8000/api/agent/dashboard/` | `GET` | Interactive Dark-Mode Visual Dashboard |
| `http://127.0.0.1:8000/api/agent/run/` | `POST` | Execute an autonomous agent task with full trace |
| `http://127.0.0.1:8000/api/model/status/` | `GET` | GGUF model detection & Metal GPU runtime status |
| `http://127.0.0.1:8000/api/model/load/` | `POST` | Load / switch GGUF model in Metal GPU unified memory |
| `http://127.0.0.1:8000/api/agent/tools/` | `GET` | List all registered tools and parameter schemas |
| `http://127.0.0.1:8000/api/agent/memory/` | `GET / DELETE` | Inspect or clear session memory & history |
| `http://127.0.0.1:8000/api/status/` | `GET` | System health, GPU accelerator & Agentic AI status |
| `http://127.0.0.1:8000/api/benchmark/` | `POST` | Live PyTorch tensor calculation on Apple Silicon MPS |
| `http://127.0.0.1:8000/api/predict/` | `POST` | Single-shot NLP sentiment inference endpoint |

---

## 🧪 Running Automated Tests
```bash
source ~/.global-python-env/bin/activate
python manage.py test
```
