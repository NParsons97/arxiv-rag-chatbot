"""
arXiv RAG Chatbot — Web Server
Run with: uvicorn server:app --reload
"""

import re
import json
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import arxiv
import chromadb
import httpx
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

OLLAMA_MODEL = "llama3.2:1b"  # faster; swap to "llama3.2" for higher quality
MAX_PAPERS = 3               # fewer papers = faster indexing
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("arxiv_papers")
indexed_papers = {}
conversation_history = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the model so the first user request isn't slow
    try:
        ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}])
    except Exception:
        pass
    yield


# ── RAG helpers (same logic as arxiv_rag.py) ────────────────────────────────

def chunk_text(text: str) -> "list[str]":
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
    return chunks


def fetch_full_text(paper: arxiv.Result) -> Optional[str]:
    try:
        import fitz
        pdf_url = next((l.href for l in paper.links if l.title == "pdf"), None) or paper.pdf_url
        response = httpx.get(pdf_url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return None


def index_paper(paper: arxiv.Result) -> str:
    paper_id = paper.get_short_id()
    if paper_id in indexed_papers:
        return paper_id

    abstract = paper.summary.strip()
    text = f"Title: {paper.title}\n\nAuthors: {', '.join(str(a) for a in paper.authors)}\n\nAbstract:\n{abstract}"

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{paper_id}::chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "paper_id": paper_id,
            "title": paper.title,
            "authors": ", ".join(str(a) for a in paper.authors[:5]),
            "published": str(paper.published.date()),
            "url": paper.entry_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    indexed_papers[paper_id] = {
        "title": paper.title,
        "authors": ", ".join(str(a) for a in paper.authors[:5]),
        "published": str(paper.published.date()),
        "url": paper.entry_id,
        "abstract": abstract,
    }
    return paper_id


def search_and_index(query: str, category: Optional[str] = None) -> "list[str]":
    full_query = f"cat:{category} AND {query}" if category else query
    search = arxiv.Search(
        query=full_query,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return [index_paper(p) for p in search.results()]


def retrieve_context(query: str, n_results: int = 8):
    """Returns (context_string, sources_list) where sources_list has one entry per paper."""
    if collection.count() == 0:
        return "", []
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )
    seen = {}   # paper_id -> source entry
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        pid = meta["paper_id"]
        if pid not in seen:
            seen[pid] = {
                "title": meta["title"],
                "authors": meta["authors"],
                "published": meta["published"],
                "url": meta["url"],
                "excerpts": [],
            }
            parts.append(
                f"--- Paper: {meta['title']} ({meta['published']}) ---\n"
                f"Authors: {meta['authors']}\nURL: {meta['url']}\n\n{doc}"
            )
        else:
            parts.append(f"--- (continued: {meta['title']}) ---\n{doc}")
        # Keep the two most relevant excerpts per paper
        if len(seen[pid]["excerpts"]) < 2:
            seen[pid]["excerpts"].append(doc[:400].strip())
    return "\n\n".join(parts), list(seen.values())


def build_system_prompt() -> str:
    paper_list = ""
    if indexed_papers:
        paper_list = "\n\nCurrently indexed papers:\n" + "\n".join(
            f"- {info['title']} ({info['published']})" for info in indexed_papers.values()
        )
    return (
        "You are an expert research assistant that helps users understand scientific papers from arXiv. "
        "When answering:\n"
        "- Explain concepts clearly, avoiding unnecessary jargon — assume a curious non-expert\n"
        "- Cite specific papers by title when referencing their content\n"
        "- If the context is insufficient, say so and suggest what to search for\n"
        "- Use markdown formatting for clarity\n"
        "- For complex topics, break down the explanation step by step"
        + paper_list
    )


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    category: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.get("/papers")
async def get_papers():
    return list(indexed_papers.values())


@app.post("/search")
async def search(req: SearchRequest):
    loop = asyncio.get_event_loop()
    ids = await loop.run_in_executor(None, search_and_index, req.query, req.category)
    return {"indexed": len(ids), "papers": [indexed_papers[i] for i in ids if i in indexed_papers]}


@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message
    category = req.category

    search_triggers = re.compile(
        r"\b(search|find|look up|fetch|get papers? (on|about)|papers? on|arxiv)\b",
        re.IGNORECASE,
    )

    async def generate() -> AsyncGenerator[str, None]:
        # Always search arXiv for each message so each question gets its own fresh papers
        query = re.sub(
            r"^(search for|find papers? (on|about)|get papers? (on|about)|look up|fetch)\s+",
            "",
            user_message,
            flags=re.IGNORECASE,
        ).strip()
        cat_label = f" in {category}" if category else ""
        yield f"data: {json.dumps({'type': 'status', 'text': f'Searching arXiv{cat_label}...'})}\n\n"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, search_and_index, query, category)
        yield f"data: {json.dumps({'type': 'status', 'text': f'Indexed {len(indexed_papers)} paper(s) total — generating answer...'})}\n\n"

        context, sources = retrieve_context(user_message)
        augmented = (
            f"<retrieved_context>\n{context}\n</retrieved_context>\n\nUser question: {user_message}"
            if context else user_message
        )

        conversation_history.append({"role": "user", "content": augmented})
        messages = [{"role": "system", "content": build_system_prompt()}] + conversation_history

        full_response = ""
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        for chunk in stream:
            text = chunk["message"]["content"]
            full_response += text
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

        conversation_history[-1] = {"role": "user", "content": user_message}
        conversation_history.append({"role": "assistant", "content": full_response})
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/clear")
async def clear():
    conversation_history.clear()
    return {"ok": True}
