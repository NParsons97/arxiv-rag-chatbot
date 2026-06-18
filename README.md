# arXiv RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that searches arXiv for scientific papers and explains them in plain language. Runs 100% locally for free using Ollama.

## How it works

1. You ask a question or name a topic
2. The app searches arXiv and retrieves relevant papers
3. It chunks and embeds the papers into a local vector store (ChromaDB)
4. Your question + the most relevant excerpts are sent to a local LLM (via Ollama)
5. The LLM explains the research in plain language

## Requirements

- [Ollama](https://ollama.com) installed and running
- Python 3.9+

## Setup

**1. Install Ollama and pull the model**
```bash
# Download from https://ollama.com, then:
ollama pull llama3.2
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the web app**
```bash
uvicorn server:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**Or run the terminal version**
```bash
python arxiv_rag.py
```

## Stack

- **LLM**: [Ollama](https://ollama.com) (llama3.2) — local, free, no API key
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store**: ChromaDB (in-memory)
- **Paper source**: [arXiv](https://arxiv.org) via the `arxiv` Python library
- **Web server**: FastAPI + uvicorn
- **Frontend**: Vanilla HTML/CSS/JS with streaming responses
