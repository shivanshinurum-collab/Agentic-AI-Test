# AI Model & Real Agentic AI Backend (Django + PyTorch + Qwen3-4B GGUF + Live Multi-Platform Tools)

This repository contains a high-performance Django backend integrated with PyTorch, Apple Silicon GPU acceleration (Metal / MPS), and a **Real Agentic AI** framework powered by **Qwen3-4B-Q4_K_M.gguf** and live multi-platform external APIs with **Zero Hardcoded Data**.

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

## 🧠 Real Agentic AI Architecture (Zero Hardcoded Data)

Unlike static rule engines with hardcoded datasets, this system executes **autonomous multi-step ReAct (Reasoning + Action + Observation)** loops that query **real-time live platforms and APIs** dynamically on demand:

```
[User Objective] 
      │
      ▼
┌──────────────┐    Reasoning    ┌──────────────────────────────────────────────┐
│ ReAct Loop   │ ──────────────> │ Qwen3-4B-Q4_K_M.gguf (Metal GPU Accelerated) │
│ Orchestrator │ <────────────── │ (Thought + Action Plan)                      │
└──────┬───────┘                 └──────────────────────────────────────────────┘
       │
       ▼ Action (Live Tool Invocation)
┌─────────────────────────────────────────────────────────────────────────┐
│ Pluggable Live Tool Registry (Zero Hardcoding)                          │
│  ├─ 🗺️ Google Maps & OSRM Engine (Live Driving Routes, Turns & Geocoding)│
│  ├─ 🌦️ Open-Meteo Weather (Real-Time Temperatures & 5-Day Forecasts)    │
│  ├─ 🌐 DuckDuckGo Web Search (Live Internet Search & Reference Links)   │
│  ├─ 📚 Wikipedia REST API (Real-Time Encyclopedia Knowledge & Sights)   │
│  ├─ 🔍 Google Places & OSM POI (Live Hotels, Cafes & Attraction Search) │
│  ├─ 📄 Web Page Content Reader (Live URL Scraping & Markdown Extractor) │
│  ├─ 🧭 Dynamic Trip Orchestrator (Multi-Tool Live Travel Synthesizer)   │
│  ├─ 🧮 Calculator (Safe AST Math & Statistical Formulas)                │
│  ├─ 🐍 Python Interpreter (Sandboxed Code Execution)                     │
│  ├─ 🧠 NLP Analyzer (PyTorch Apple Silicon MPS Sentiment Analysis)      │
│  ├─ ⏰ DateTime Tool (Temporal Calculations & ISO Timestamps)           │
│  └─ 💾 Memory Store (Session-Aware Scratchpad Retention)                │
└─────────────────────────────────────────────────────────────────────────┘
       │ Observation
       ▼
[Self-Reflection & Dynamic Multi-Pass Synthesis]
```

---

## 🛠️ Live External Platform Connectors

### 1. 🗺️ Google Maps & OSRM Routing Engine
- **Google Maps API**: Real-time directions, traffic, geocoding, and place ratings when `GOOGLE_MAPS_API_KEY` is present.
- **Live OSRM Fallback**: High-precision road routing (`router.project-osrm.org`) providing exact road distances in km, driving/transit duration, and turn-by-turn maneuvers worldwide.
- **OpenStreetMap Nominatim**: Global live geocoding and reverse geocoding for any address or coordinate.

### 2. 🌦️ Open-Meteo Weather API
- 100% free, real-time meteorological API providing current live temperature (°C / °F), apparent temperature, humidity, wind speeds, WMO weather condition badges, and 5-day daily forecasts.

### 3. 🌐 DuckDuckGo Live Web Search
- Real-time internet search fetching live snippets, page titles, domain references, and URLs for any current event or topic.

### 4. 📚 Wikipedia REST API
- Real-time entity lookups, historical context, cultural details, landmark summaries, and coordinates across the globe.

### 5. 🔍 Places & Point of Interest (POI) Discovery
- Live Google Places & OpenStreetMap POI discovery for top-rated restaurants, luxury/budget hotels, cafes, temples, and attractions in any city.

### 6. 🧭 Dynamic Trip & Travel Orchestrator
- Dynamically coordinates Google Maps, Open-Meteo, Wikipedia, Google Places, and AST Calculator in real-time to generate complete master travel guides for **any destination in the world** without static templates.

---

## 🦙 GGUF Local Model Integration (`Qwen3-4B-Q4_K_M.gguf`)

The backend is fully configured to execute `Qwen3-4B-Q4_K_M.gguf` with **Apple Silicon Metal GPU offloading** (`n_gpu_layers=-1`):

1. **Option A (Recommended)**: Drop `Qwen3-4B-Q4_K_M.gguf` inside the [`models/`](file:///Users/inurum/Documents/AI_Model/models/) folder:
   ```bash
   cp /path/to/your/Qwen3-4B-Q4_K_M.gguf /Users/inurum/Documents/AI_Model/models/
   ```
2. **Option B (Direct File Path)**: Provide any model file path directly via the Dashboard UI or API payload:
   ```json
   {
     "prompt": "Plan a 2-day trip from Bangalore to Coorg with live weather",
     "planner_mode": "gguf",
     "model_path": "/path/to/your/Qwen3-4B-Q4_K_M.gguf"
   }
   ```

---

## 🛠️ Quick Start

### 1. Run the Development Server
```bash
source ~/.global-python-env/bin/activate
python manage.py runserver 8000
```

### 2. Open the ChatGPT-Style Conversational UI & Dashboard
- **Chat Assistant**: 👉 **`http://127.0.0.1:8000/`** (or `http://127.0.0.1:8000/chat/`)
- **Agent Control Dashboard**: 👉 **`http://127.0.0.1:8000/api/agent/dashboard/`**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `http://127.0.0.1:8000/chat/` | `GET` | ChatGPT-Style Agentic Conversational Interface |
| `http://127.0.0.1:8000/api/agent/dashboard/` | `GET` | Interactive Dark-Mode Control Dashboard |
| `http://127.0.0.1:8000/api/agent/run/` | `POST` | Execute an autonomous agent task with live tool execution & full trace |
| `http://127.0.0.1:8000/api/agent/tools/` | `GET` | List all 11+ live registered tools and schemas |
| `http://127.0.0.1:8000/api/agent/memory/` | `GET / DELETE` | Inspect or clear session memory & history |
| `http://127.0.0.1:8000/api/status/` | `GET` | System health, GPU accelerator & Agentic AI status |
| `http://127.0.0.1:8000/api/model/status/` | `GET` | GGUF model detection & Metal GPU runtime status |
| `http://127.0.0.1:8000/api/model/load/` | `POST` | Load / switch GGUF model in Metal GPU memory |
| `http://127.0.0.1:8000/api/benchmark/` | `POST` | Live PyTorch tensor calculation on Apple Silicon MPS |
| `http://127.0.0.1:8000/api/predict/` | `POST` | Single-shot NLP sentiment inference endpoint |

---

## 🧪 Running Automated Tests
```bash
source ~/.global-python-env/bin/activate
python manage.py test ai_service
```
