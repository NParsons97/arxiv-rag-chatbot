"""
arXiv RAG Chatbot
Retrieves arXiv papers and uses Ollama (local, free) to answer questions.
Install Ollama: https://ollama.com
Then run: ollama pull llama3.2
"""

import re
import sys
from typing import Optional
import arxiv
import chromadb
import httpx
import ollama
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

OLLAMA_MODEL = "llama3.2"
MAX_PAPERS = 5
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

console = Console()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("arxiv_papers")

conversation_history = []
indexed_papers = {}  # type: dict


def chunk_text(text: str) -> "list[str]":
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
    return chunks


def fetch_abstract(paper: arxiv.Result) -> str:
    return paper.summary.strip()


def fetch_full_text(paper: arxiv.Result) -> Optional[str]:
    """Download PDF and extract text via PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        pdf_url = next(
            (l.href for l in paper.links if l.title == "pdf"), None
        )
        if not pdf_url:
            pdf_url = paper.pdf_url

        response = httpx.get(pdf_url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return None


def index_paper(paper: arxiv.Result, use_full_text: bool = False) -> str:
    paper_id = paper.get_short_id()
    if paper_id in indexed_papers:
        return paper_id

    text = None
    if use_full_text:
        with console.status("[dim]Downloading PDF...[/dim]"):
            text = fetch_full_text(paper)

    if not text:
        text = f"Title: {paper.title}\n\nAuthors: {', '.join(str(a) for a in paper.authors)}\n\nAbstract:\n{fetch_abstract(paper)}"

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
        "abstract": fetch_abstract(paper),
    }

    return paper_id


def search_and_index(query: str, max_results: int = MAX_PAPERS) -> "list[str]":
    console.print(f"\n[dim]Searching arXiv for:[/dim] [italic]{query}[/italic]")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    paper_ids = []
    with console.status("[dim]Fetching papers...[/dim]"):
        for paper in search.results():
            pid = index_paper(paper)
            paper_ids.append(pid)
            console.print(f"  [green]✓[/green] {paper.title[:80]}...")

    return paper_ids


def retrieve_context(query: str, n_results: int = 8) -> str:
    if collection.count() == 0:
        return ""

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )

    seen_papers: set[str] = set()
    context_parts = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        pid = meta["paper_id"]
        if pid not in seen_papers:
            seen_papers.add(pid)
            context_parts.append(
                f"--- Paper: {meta['title']} ({meta['published']}) ---\n"
                f"Authors: {meta['authors']}\nURL: {meta['url']}\n\n{doc}"
            )
        else:
            context_parts.append(f"--- (continued: {meta['title']}) ---\n{doc}")

    return "\n\n".join(context_parts)


def build_system_prompt() -> str:
    paper_list = ""
    if indexed_papers:
        paper_list = "\n\nCurrently indexed papers:\n" + "\n".join(
            f"- [{pid}] {info['title']} ({info['published']})"
            for pid, info in indexed_papers.items()
        )

    return (
        "You are an expert research assistant that helps users understand scientific papers from arXiv. "
        "You have access to a retrieval system that surfaces relevant excerpts from indexed papers. "
        "When answering:\n"
        "- Explain concepts clearly, avoiding unnecessary jargon — assume a curious non-expert\n"
        "- Cite specific papers by title when referencing their content\n"
        "- If the context is insufficient, say so and suggest what to search for\n"
        "- Use markdown formatting for clarity\n"
        "- For complex topics, break down the explanation step by step"
        + paper_list
    )


def chat(user_message: str) -> None:
    # Detect if the user wants to search arXiv
    search_triggers = re.compile(
        r"\b(search|find|look up|fetch|get papers? (on|about)|papers? on|arxiv)\b",
        re.IGNORECASE,
    )
    is_explicit_search = bool(search_triggers.search(user_message))

    # Auto-search if no papers indexed yet, or user explicitly asks to search
    if not indexed_papers or is_explicit_search:
        # Extract search query: strip common prefixes
        query = re.sub(
            r"^(search for|find papers? (on|about)|get papers? (on|about)|look up|fetch)\s+",
            "",
            user_message,
            flags=re.IGNORECASE,
        ).strip()
        search_and_index(query)

    # Retrieve relevant context
    context = retrieve_context(user_message)

    # Compose the user turn with injected context
    augmented_message = user_message
    if context:
        augmented_message = (
            f"<retrieved_context>\n{context}\n</retrieved_context>\n\n"
            f"User question: {user_message}"
        )

    conversation_history.append({"role": "user", "content": augmented_message})

    console.print()

    # Stream the response via Ollama
    full_response = ""
    messages = [{"role": "system", "content": build_system_prompt()}] + conversation_history

    console.print("[bold cyan]Assistant:[/bold cyan]")
    stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
    for chunk in stream:
        text = chunk["message"]["content"]
        console.print(text, end="")
        full_response += text

    console.print("\n")

    # Store assistant response (without context injection for cleaner history)
    conversation_history[-1] = {"role": "user", "content": user_message}
    conversation_history.append({"role": "assistant", "content": full_response})


def show_indexed_papers() -> None:
    if not indexed_papers:
        console.print("[dim]No papers indexed yet.[/dim]")
        return
    console.print("\n[bold]Indexed Papers:[/bold]")
    for pid, info in indexed_papers.items():
        console.print(f"  [cyan]{pid}[/cyan] — {info['title']} ({info['published']})")
        console.print(f"    [dim]{info['url']}[/dim]")
    console.print()


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]arXiv RAG Chatbot[/bold cyan]\n"
            "[dim]Powered by Ollama (local, free) + arXiv[/dim]\n\n"
            "Commands:\n"
            "  [green]/search <query>[/green]  — search and index arXiv papers\n"
            "  [green]/papers[/green]          — list indexed papers\n"
            "  [green]/clear[/green]           — clear conversation history\n"
            "  [green]/quit[/green]            — exit",
            title="Welcome",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)

        if user_input.lower() == "/papers":
            show_indexed_papers()
            continue

        if user_input.lower() == "/clear":
            conversation_history.clear()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            search_and_index(query)
            console.print(f"[dim]Indexed {len(indexed_papers)} paper(s) total.[/dim]")
            continue

        chat(user_input)


if __name__ == "__main__":
    main()
