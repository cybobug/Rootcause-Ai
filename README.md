# 🔍 RootCause AI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](#-getting-started)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)

> **RootCause AI** is an **LLM-powered incident analysis and root cause detection engine** that combines  
real-time anomaly detection, causal chain reasoning, and smart recommendations.  
It leverages **GPT-OSS-120B (Cerebras / HuggingFace)** and **Streamlit** for an interactive UI.

---

## ✨ Features

- 🤖 **LLM Reasoning Pipeline**  
  Uses **GPT-OSS-120B** or OpenAI models for structured incident diagnosis (root causes, causal chain, mitigations).

- 📊 **Real-Time Analytics Engine**  
  Detects anomalies, predicts failures, and calculates system health scores using statistical trend analysis.

- 🔗 **Multi-Source Connectors**  
  Ingests events from:
  - Logs (JSON, common, Apache)
  - GitHub commits
  - Metrics (Datadog, custom)
  - Bug reports

- 🧪 **Incident Simulator**  
  Generates realistic failure scenarios (DB deadlocks, API cascade failures, memory leaks, config errors).

- 🎨 **Premium Streamlit UI**  
  - Live anomaly detection dashboards  
  - Causal chain visualizations (Plotly + NetworkX)  
  - AI chat interface for natural queries  
  - Gamified **achievements system**

- 🔐 **Flexible API Config**  
  Switch between **OpenAI** and **Hugging Face** backends directly from the UI.

---

## 📂 Project Structure

Rootcause-Ai/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py              # Main Streamlit application entrypoint
└── rootcause_ai/
    ├── __init__.py
    ├── analyzer.py     # Core analysis engine and LLM pipeline
    ├── connectors.py   # Data ingestion connectors
    ├── simulator.py    # Incident simulation engine
    └── utils.py        # Utility functions (e.g., graph visualization)


---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/rootcause_ai.git
cd rootcause_ai
 
### 2️⃣ Create & activate a virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Run the Streamlit app
streamlit run app.py
or
python -m streamlit run app.py

## ⚙️ Configuration

RootCause AI requires API keys for LLMs and connectors.
You can configure them in the Streamlit sidebar or via environment variables:
# LLM
export OPENAI_API_KEY="your-openai-key"
export HF_API_TOKEN="your-huggingface-token"

# GitHub
export GITHUB_TOKEN="your-github-token"

# Datadog
export DD_API_KEY="your-datadog-key"
export DD_APP_KEY="your-datadog-app-key"

In the UI, you can also switch provider between:

openai (default GPT-4 / GPT-4o models)
huggingface (openai/gpt-oss-120b:cerebras)

🧪 Example Usage

Simulate an incident:

Choose a scenario (e.g., database deadlock)

Generate events (logs, metrics, bug reports)

Run AI analysis for root cause and recommendations

Upload logs / metrics:
Ingest your system data and let the AI build causal chains.

Ask in natural language:

“What caused the API slowdown?”
“Show me recent anomalies.”
“Predict memory usage trends.”

📜 License

This project is licensed under the Apache License 2.0.
See LICENSE
🌟 Star this Repo

If you find this project useful, consider starring ⭐ the repo to help others discover it!

🤝 Contributing

Contributions are welcome!
Feel free to open an issue or pull request to improve RootCause AI.

📬 Contact

Author: Garvit Haswani
