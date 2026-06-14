# 🚗 Multi-Agent Dealership Assistant

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Streamlit](https://img.shields.io/badge/ui-Streamlit-ff4b4b)
![OpenAI](https://img.shields.io/badge/ai-OpenAI-black)
![RAG](https://img.shields.io/badge/RAG-FAISS-purple)

A professional **multi-agent conversational AI assistant** for car dealerships.  
It combines dealer-specific knowledge base search, natural-language booking flows, OpenAI-powered reasoning, and a polished Streamlit frontend.

---

## ✨ Highlights

- 🧠 **Multi-agent orchestration** for routing user messages to the right workflow.
- 📚 **Dealer-specific RAG** with isolated FAISS knowledge bases per dealership.
- 🚘 **Automotive Q&A** grounded only in uploaded dealer documents.
- 🗓️ **Service booking flow** that collects one field at a time.
- 🧪 **Test drive booking flow** with configurable form fields.
- 🕒 **Natural date/time understanding** for inputs like `next week Sunday`, `20 June`, and `8pm`.
- 🧾 **Config-driven intents and forms** using JSON files.
- 🧊 **Liquid glass dark UI** with gradient styling in Streamlit.
- 🔐 **Safe environment setup** using `.env.example`; real `.env` is ignored by git.

---

## 📸 What The App Does

The assistant lets a user select a dealership and chat with an AI agent that can:

- Answer questions from the selected dealer's documents.
- Avoid mixing knowledge between different dealerships.
- Start a service booking.
- Start a test drive booking.
- Capture structured form details.
- Hand off to a human agent when needed.
- Show agent trace logs for debugging.

Example KB questions:

- `When was Charles Hurst founded?`
- `What types of vehicles does Charles Hurst sell?`
- `Does Charles Hurst offer apprenticeships?`

Example booking conversation:

```text
User: I want to book a service
Assistant: What is the make and model of your vehicle?

User: BMW 320d
Assistant: What date would you like to book your service?

User: next week Sunday
Assistant: What time would you prefer?

User: 8pm
Assistant: What name should we put the booking under?
```

---

## 🤖 Agents

### 1. 🧭 Orchestrator Agent

Routes every user message to the correct workflow:

- General dealership enquiry
- Service booking
- Test drive booking
- Human handoff

It decides which specialist agent should handle the next step.

### 2. 🎯 Intent Classification Agent

Classifies the user's message into supported intents:

- `general_enquiry`
- `service_booking`
- `test_drive`
- `human_handoff`

The intent configuration lives in:

```text
configs/intents.json
```

### 3. 📚 Knowledge Agent

Answers dealer-specific questions using Retrieval-Augmented Generation.

Important behavior:

- Uses the selected dealer only.
- Searches the selected dealer's FAISS index.
- Answers from retrieved chunks.
- Falls back to `"This information is not available in the knowledge base."` when the answer is not present.
- Supports direct matching for numbered Q&A documents.

### 4. 📝 Booking Agent

Collects booking form fields one at a time.

Supported booking types:

- Service booking
- Test drive booking

Forms live in:

```text
configs/forms/
```

### 5. ✅ QA / Validation Logic

Normalizes and validates user-provided booking fields.

Examples:

- `next week sunday` → `Sunday, 21 June 2026`
- `20 june` → `Saturday, 20 June 2026`
- `8pm` → `8:00 PM`

The system uses GPT-backed normalization with deterministic fallbacks.

### 6. 🙋 Human Handoff Agent

Handles requests to speak with a human agent and captures lead details when needed.

---

## 🧠 RAG Architecture

Each dealer has isolated knowledge files.

```text
Data/
├── Charles_Hurst_20_QA.pdf
├── Lookers_20_QA.pdf
├── Sytner_20_QA.pdf
├── updated_car_QA_with_appointment.pdf
├── Charles_Hurst/
│   ├── kb.index          # generated, ignored by git
│   └── kb_chunks.json    # generated, ignored by git
├── Lookers/
├── Sytner/
└── updated_car/
```

The source PDFs are committed. Generated FAISS files are ignored and can be rebuilt from the sidebar.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** for frontend
- **OpenAI API** for GPT and embeddings
- **OpenAI Agents SDK** for agent workflows
- **FAISS** for vector similarity search
- **pdfplumber / PyMuPDF** for PDF extraction
- **python-dotenv** for environment variables

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/thisisahmad/multi-agent-dealership-assistant.git
cd multi-agent-dealership-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
OPENAI_TEMPORAL_MODEL=gpt-5
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 📚 Rebuilding Knowledge Bases

In the sidebar:

1. Select a dealer, for example `Charles_Hurst`.
2. Click **Rebuild KB for Charles_Hurst**.
3. The app creates a dealer-specific FAISS index.

Generated files are intentionally ignored by git:

```text
Data/<dealer>/kb.index
Data/<dealer>/kb_chunks.json
```

---

## ⚙️ Configuration

### Intents

```text
configs/intents.json
```

Defines supported user intents and keyword mappings.

### Forms

```text
configs/forms/service_booking.json
configs/forms/test_drive.json
```

Defines required fields, prompts, validation rules, and confirmation templates.

### Behavior Rules

```text
configs/behaviour_rules.json
```

Defines strict RAG behavior and response guidelines.

### Dealer Config

```text
configs/dealer_config.json
```

Stores dealer isolation and knowledge base settings.

---

## 🧪 Testing

A lightweight system test file is included:

```bash
python test_refactored_system.py
```

You can also manually test in the Streamlit chat:

```text
When was Charles Hurst founded?
What types of vehicles does Charles Hurst sell?
I want to book a service
next week Sunday
8pm
```

---

## 🔐 Security Notes

- Never commit `.env`.
- `.env` is ignored by `.gitignore`.
- Use `.env.example` to document required environment variables.
- Generated local databases and vector indexes are ignored.

---

## 🗺️ Roadmap

- 🌍 Multi-language dealer assistant
- 📅 Real calendar integration
- 📞 CRM / lead management integration
- 📊 Admin dashboard for conversations and bookings
- 🔎 Hybrid search with metadata filters
- 🧪 Expanded automated test coverage

---

## 📦 Version

Current version: **v1.0.0**

Initial release includes:

- Dealer-specific RAG
- Multi-agent routing
- Service and test drive booking
- Natural language date/time normalization
- Dark glass Streamlit UI
- Config-driven intents and forms

---

## 👤 Author

Built by [Muhammad Ahmad](https://github.com/thisisahmad).

---

## 📄 License

This project is currently provided as a private/prototype implementation.  
Add a license file before publishing it as an open-source project.
