"""
knowledge_retrieval_agent + rag_validation_agent (Section 7).
Sequential: retrieve, then independently validate that retrieval actually
supports the claims before it's trusted downstream (Principle 3, "do not
trust a single source" -- extended here to "do not trust retrieval blindly").

RAG backend is switchable via config.settings.RAG_BACKEND:
  - "local"  (default): our TF-IDF tool over sample_documents/. Works with
    just GOOGLE_API_KEY, no GCP project required.
  - "vertex": Google's managed Vertex AI RAG Engine (VertexAiRagRetrieval).
    Requires GOOGLE_GENAI_USE_VERTEXAI=TRUE, a real GCP project, Application
    Default Credentials, and an existing RAG corpus in RAG_CORPUS_RESOURCE_NAME.
    NOTE (ADK constraint, verified against installed version): VertexAiRagRetrieval
    must be the ONLY tool on its agent -- it cannot be mixed with other tools
    on the same LlmAgent. That's why it gets its own agent below with nothing
    else attached, rather than sharing knowledge_retrieval_agent's tool list.
"""
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from tools.rag_tools import rag_search, rerank_documents
from tools.evidence_tools import validate_evidence
from config import settings

rag_search_tool = FunctionTool(rag_search)
rerank_tool = FunctionTool(rerank_documents)
validate_evidence_tool = FunctionTool(validate_evidence)

MODEL = "gemini-2.5-flash"

_RETRIEVAL_INSTRUCTION = """Business problem: {{business_problem?}}
Process map: {{process_map?}}

{tool_guidance}
Preserve document/section/page metadata (or corpus/chunk id for Vertex
results) for every chunk you use -- never restate a chunk's content without
its source attribution.

CRITICAL: judge relevance honestly. If the retrieved chunks are about a
different company, industry, or process than the business problem above (or
if the corpus returns nothing, or only marginally related material), say so
explicitly and plainly -- e.g. "No relevant internal documents were found for
this business problem; the corpus contains [X] instead." Do NOT paraphrase
irrelevant retrieved content into something that sounds applicable. A
confident-sounding answer built on the wrong documents is worse than an
honest "no relevant internal evidence found" (Section 26: never fabricate
missing information). Output the retrieved chunks and a short synthesis,
clearly labeled as INTERNAL evidence."""

if settings.RAG_BACKEND == "vertex":
    try:
        from vertexai.preview import rag
        from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "STRATEGIST_RAG_BACKEND=vertex requires the 'google-cloud-aiplatform' package "
            "(provides the vertexai module). Install it with: "
            "pip install google-cloud-aiplatform>=1.60.0"
        ) from e

    if not settings.RAG_CORPUS_RESOURCE_NAME:
        raise RuntimeError(
            "STRATEGIST_RAG_BACKEND=vertex but RAG_CORPUS_RESOURCE_NAME is not set. "
            "Create a RAG corpus first (see README 'Switching to Vertex AI RAG Engine') "
            "and set its resource name, e.g. projects/P/locations/L/ragCorpora/C."
        )

    vertex_rag_tool = VertexAiRagRetrieval(
        name="vertex_rag_retrieval",
        description="Retrieves relevant internal enterprise document chunks from the Vertex AI RAG corpus.",
        rag_resources=[rag.RagResource(rag_corpus=settings.RAG_CORPUS_RESOURCE_NAME)],
        similarity_top_k=8,
        vector_distance_threshold=0.6,
    )

    knowledge_retrieval_agent = LlmAgent(
        name="knowledge_retrieval_agent",
        model=MODEL,
        description="Retrieves relevant internal enterprise documents via Vertex AI RAG Engine.",
        instruction=_RETRIEVAL_INSTRUCTION.format(
            tool_guidance="Call `vertex_rag_retrieval` with focused queries (one call per "
            "distinct sub-topic -- e.g. current process steps, staffing, systems in use, "
            "applicable governance policy)."
        ),
        # VertexAiRagRetrieval must be the only tool on this agent -- ADK limitation.
        tools=[vertex_rag_tool],
        output_key="internal_evidence",
    )
else:
    knowledge_retrieval_agent = LlmAgent(
        name="knowledge_retrieval_agent",
        model=MODEL,
        description="Retrieves relevant internal enterprise documents (local TF-IDF RAG).",
        instruction=_RETRIEVAL_INSTRUCTION.format(
            tool_guidance="Call `rag_search` with focused queries (one call per distinct "
            "sub-topic -- e.g. current process steps, staffing, systems in use, applicable "
            "governance policy) to retrieve relevant internal document chunks. Use "
            "`rerank_documents` if you gather chunks from multiple queries and need a "
            "single relevance ordering."
        ),
        tools=[rag_search_tool, rerank_tool],
        output_key="internal_evidence",
    )

rag_validation_agent = LlmAgent(
    name="rag_validation_agent",
    model=MODEL,
    description="Checks whether retrieved internal evidence actually supports the claims made from it.",
    instruction="""Retrieved internal evidence: {internal_evidence?}

List the specific claims the retrieval agent drew from these chunks, then call
`validate_evidence` with those claims and the evidence records to check
coverage. Do NOT treat retrieved documents as automatically authoritative
(Section 7) -- flag any claim that outruns what the chunk text actually says,
and flag any low similarity_score chunks (<0.15, when available) as weak support.""",
    tools=[validate_evidence_tool],
    output_key="internal_evidence_validated",
)

knowledge_retrieval_pipeline = SequentialAgent(
    name="knowledge_retrieval_pipeline",
    description="Retrieve internal evidence via RAG (local or Vertex), then validate it before it's trusted downstream.",
    sub_agents=[knowledge_retrieval_agent, rag_validation_agent],
)
