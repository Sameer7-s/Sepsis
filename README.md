---
title: Sepsis AI Agent
emoji: 🏥
colorFrom: pink
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: AI-powered Sepsis AI Agent with clinical decision intelligence
---

<div align="center">

<pre style="font-size:9px; line-height:1.05; margin:0;">
███████╗███████╗██████╗ ███████╗██╗███████╗
██╔════╝██╔════╝██╔══██╗██╔════╝██║██╔════╝
███████╗█████╗  ██████╔╝███████╗██║███████╗
╚════██║██╔══╝  ██╔═══╝ ╚════██║██║╚════██║
███████║███████╗██║     ███████║██║███████║
╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝╚══════╝

    █████╗ ██╗ █████╗  ██████╗ ███████╗███╗ ██╗████████╗
   ██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝████╗██║╚══██╔══╝
   ███████║██║███████║██║  ███╗█████╗  ██╔████║   ██║
   ██╔══██║██║██╔══██║██║   ██║██╔══╝  ██║╚███║   ██║
   ██║  ██║██║██║  ██║╚██████╔╝███████╗██║ ╚██║   ██║
   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝
</pre>

**AI-Powered Clinical Decision Engine · OpenEnv Evaluation Agent**

![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-Integrated-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)

</div>

---

## 🧬 Overview

**Sepsis AI Agent** is an evaluation-ready clinical decision system designed for simulating sepsis management within an OpenEnv-compatible environment.

Sepsis causes **11 million deaths annually**, many preventable with timely intervention.  
This project focuses on modeling **decision-making under uncertainty**, combining learned reasoning with deterministic safety constraints.

### Core Components

- 🧠 LLM-based reasoning via API proxy  
- ⚙️ Deterministic fallback for reliability  
- 🏥 FastAPI-based simulation backend  
- 🔁 OpenEnv-compatible interaction loop  

---

## 🏗️ System Architecture

```text
inference.py (Agent)
        │
        ▼
Decision Layer
├── LLM (API_BASE_URL)
└── Rule Engine
        │
        ▼
backend_api.py (FastAPI)
        │
        ▼
Sepsis Environment
        │
        ▼
Scenarios (early / severe / shock)
```

---

## 🤖 Decision Engine

### LLM Reasoning

```python
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)
```

- Deterministic (`temperature = 0`)  
- Validator-compliant proxy usage  
- Context-aware clinical reasoning  

---

### Rule-Based Fallback

Ensures consistent execution even if LLM fails:

| Condition | Action |
|----------|--------|
| No antibiotics | administer_antibiotics |
| SpO₂ < 90 | oxygen_therapy |
| MAP < 65 | start_vasopressors |
| MAP < 70 / Lactate high | give_fluids |
| Stable | observe |

---

## 🏥 Simulation

| Scenario | Focus |
|----------|------|
| early_sepsis | Early intervention |
| severe_sepsis | Organ dysfunction |
| septic_shock | Critical stabilization |

---

## 📡 API

- `/health`  
- `/reset`  
- `/step`  
- `/state`  

---

## 📊 Output Format

```text
[START] task=early_sepsis
[STEP] step=1 reward=0.12
[END] task=early_sepsis score=0.73 steps=5
```

---

## 🎯 Scoring

```text
0 < score < 1
```

Strictly enforced to satisfy evaluation constraints.

---

## 🚀 Run

```bash
docker build -t sepsis-ai-agent .

docker run \
  -e API_BASE_URL=your_api \
  -e MODEL_NAME=your_model \
  -e API_KEY=your_key \
  -p 7860:7860 \
  sepsis-ai-agent
```

---

## ✅ Compliance

- ✔ Docker build  
- ✔ Structured stdout logs  
- ✔ LLM proxy usage  
- ✔ No runtime crashes  
- ✔ Score validation  

---

## 🌐 Links

- [Hugging Face Space](https://huggingface.co/spaces/Sam795/sepsis-ai-agent)  
- [GitHub Repository](https://github.com/Sameer7-s/Sepsis)  

---

## 📜 License

MIT

---

## 🚀 Hackathon Submission Support

This project is structured to support high-quality hackathon submissions, with clear system design, reproducible outputs, and evaluation-friendly behavior.

It can be extended with:

- Strong problem framing  
- Clear system architecture explanation  
- Step-by-step agent reasoning  
- Evaluation and scoring insights  

A well-structured submission often makes a significant difference in competitive environments.
