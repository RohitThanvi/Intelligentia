import os

# Safe defaults so the suite runs with zero external setup: no GCP project,
# no vertexai package, no real API key required for tests that only touch
# tools/agent_helpers.py and config/settings.py (which is most of them).
os.environ.setdefault("STRATEGIST_RAG_BACKEND", "local")
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder")
