# 🧠 Codebase Q&A Assistant

Ask natural language questions about **any public GitHub repository** using:
- **Groq Llama 3 70B** — Free LLM, ultra fast
- **HuggingFace MiniLM** — Free local embeddings
- **FAISS** — Local vector database (no paid services)
- **FastAPI** — Backend
- **Streamlit** — Frontend UI

---

## 📋 Prerequisites

- Python 3.10 or higher
- Git installed on your machine
- A free Groq API key → https://console.groq.com

---

## 🚀 Setup (Step by Step)

### 1. Unzip and enter the project folder
```bash
cd codebase-qa
```

### 2. Create a virtual environment
```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First install takes 3–5 minutes (downloads PyTorch + sentence-transformers)

### 4. Create your `.env` file
```bash
cp .env.example .env
```
Open `.env` and replace `your_groq_api_key_here` with your actual key from https://console.groq.com

### 5. Run the FastAPI backend
```bash
uvicorn app.main:app --reload
```
Backend is now running at → http://localhost:8000
Swagger docs at → http://localhost:8000/docs

### 6. Run the Streamlit UI (new terminal, same folder)
```bash
# Make sure venv is still activated!
streamlit run streamlit_app.py
```
UI is now running at → http://localhost:8501

---

## 🎯 How to Use

1. Open http://localhost:8501
2. Paste any public GitHub URL (e.g. `https://github.com/tiangolo/fastapi`)
3. Click **Process** — wait 1–3 minutes for cloning + embedding
4. Type any question about the code and click **Ask**
5. Get answers with source file references!

### Good test repositories
| Repo | Size | Good for |
|------|------|----------|
| https://github.com/tiangolo/fastapi | Medium | API questions |
| https://github.com/pallets/flask | Medium | Web framework questions |
| https://github.com/psf/requests | Small | HTTP library questions |
| Your own projects | Any | Personal showcase |

---

## 🌐 Deploying to Streamlit Cloud (Free)

### Step 1 — Push code to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/codebase-qa.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `streamlit_app.py`
5. Click **Advanced settings** → **Secrets** → paste:
```toml
GROQ_API_KEY = "your_groq_key_here"
```
6. Click **Deploy**!

### ⚠️ Important: FastAPI backend
Streamlit Cloud only hosts the UI. For the FastAPI backend you have 2 options:

**Option A — Run FastAPI locally while demoing** (easiest)
- Keep `uvicorn app.main:app --reload` running on your laptop
- Use a tunnel like [ngrok](https://ngrok.com) to expose it:
  ```bash
  ngrok http 8000
  ```
- Update `BACKEND_URL` in `streamlit_app.py` to the ngrok URL

**Option B — Deploy FastAPI on Render.com (free)**
1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
5. Add env vars: `GROQ_API_KEY`, `REPOS_DIR=/tmp/repos`, `VECTORSTORE_DIR=/tmp/vectorstores`
6. Update `BACKEND_URL` in `streamlit_app.py` to your Render URL

---

## 📁 Project Structure

```
codebase-qa/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   └── routes.py            # /upload and /ask endpoints
│   └── services/
│       ├── github_loader.py     # Clone repo + load code files
│       ├── chunking.py          # Split files into chunks
│       ├── embeddings.py        # HuggingFace MiniLM embeddings
│       ├── vector_store.py      # FAISS build + load
│       └── rag_pipeline.py      # Groq Llama 3 RAG chain
├── data/
│   ├── repos/                   # Cloned repositories (auto-created)
│   └── vectorstores/            # FAISS indexes (auto-created)
├── streamlit_app.py             # Streamlit frontend UI
├── requirements.txt
├── .env.example                 # Copy to .env and fill in keys
└── README.md
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `faiss-cpu` install fails | Run `pip install faiss-cpu --no-build-isolation` |
| `ModuleNotFoundError` | Make sure venv is activated before running |
| Backend connection refused | Start FastAPI first: `uvicorn app.main:app --reload` |
| Groq key error | Check `.env` file exists and key is correct |
| Slow first run | HuggingFace model downloads ~80MB — normal, only once |
| Private repo fails | Only public repos are supported |
| Empty answer | Try rephrasing; repo may not contain relevant code for that question |
