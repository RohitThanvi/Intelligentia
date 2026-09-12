"""
Minimal but real RAG subsystem (Section 7).

This is intentionally dependency-light (scikit-learn TF-IDF instead of a hosted
vector DB) so the project runs locally out of the box. It's a legitimate
retrieval implementation, not a stub — swap `_VECTORIZER`/`_MATRIX` for a real
vector store (e.g. Vertex AI Vector Search, Chroma, pgvector) for production
without changing the tool's public interface.

Documents are chunked, embedded (TF-IDF vectors), and retrieved by cosine
similarity. Each result preserves document/section metadata (Principle 5).
"""
import glob
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_documents")

_chunks: list[dict] = []       # [{document, section, page, text}]
_vectorizer = None
_matrix = None


def _chunk_text(text: str, doc_name: str, chunk_size: int = 500, overlap: int = 80):
    # naive section-aware split: break on markdown headers first, then by size
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    page = 1
    for section in sections:
        header_match = re.match(r"#{1,3} (.+)", section)
        section_title = header_match.group(1).strip() if header_match else "General"
        body = section
        start = 0
        while start < len(body):
            piece = body[start:start + chunk_size]
            if piece.strip():
                chunks.append({"document": doc_name, "section": section_title,
                                "page": page, "text": piece.strip()})
            start += chunk_size - overlap
            page += 1
    return chunks


def build_index() -> dict:
    """(Re)build the in-memory TF-IDF index from files in sample_documents/.

    Call this once at startup, or after adding new documents.

    Returns:
        dict with num_documents and num_chunks indexed.
    """
    global _chunks, _vectorizer, _matrix
    _chunks = []
    files = glob.glob(os.path.join(_DOCS_DIR, "*.md")) + glob.glob(os.path.join(_DOCS_DIR, "*.txt"))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        _chunks.extend(_chunk_text(text, os.path.basename(path)))

    if _chunks:
        _vectorizer = TfidfVectorizer(stop_words="english")
        _matrix = _vectorizer.fit_transform([c["text"] for c in _chunks])
    else:
        _vectorizer, _matrix = None, None

    return {"num_documents": len(files), "num_chunks": len(_chunks)}


def rag_search(query: str, top_k: int = 5) -> dict:
    """Retrieve the top-k most relevant internal document chunks for a query.

    Args:
        query: Natural-language question or claim to find supporting internal evidence for.
        top_k: Max number of chunks to return.

    Returns:
        dict with results: list of {document, section, page, text, similarity_score, source_type}.
        Empty results list (not an error) if the index has no documents yet.
    """
    if _vectorizer is None or _matrix is None or not _chunks:
        build_index()
    if _vectorizer is None or not _chunks:
        return {"results": [], "note": "No internal documents indexed yet."}

    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _matrix).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in ranked_idx:
        if sims[i] <= 0:
            continue
        c = _chunks[i]
        results.append({**c, "similarity_score": round(float(sims[i]), 3), "source_type": "internal"})
    return {"results": results}


def rerank_documents(query: str, candidate_chunks: list) -> dict:
    """Re-score a candidate list of chunks against the query for a final relevance ordering.

    Useful when combining rag_search results with web_evidence, or re-ordering
    after the knowledge_fusion step. Uses the same TF-IDF space as rag_search.

    Args:
        query: The query/claim to rerank against.
        candidate_chunks: list of dicts each containing at least a "text" field.

    Returns:
        dict with reranked: candidate_chunks sorted by relevance_score desc.
    """
    global _vectorizer
    texts = [c.get("text", "") for c in candidate_chunks]
    if not texts:
        return {"reranked": []}
    local_vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = local_vectorizer.fit_transform(texts + [query])
    except ValueError:
        return {"reranked": candidate_chunks}
    q_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    scored = [{**c, "relevance_score": round(float(s), 3)} for c, s in zip(candidate_chunks, sims)]
    scored.sort(key=lambda x: -x["relevance_score"])
    return {"reranked": scored}
