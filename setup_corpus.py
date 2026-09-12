"""
One-time (or re-run-as-needed) helper to create/populate the Vertex AI RAG
corpus used by knowledge_retrieval_agent.

Your corpus already exists:
    projects/unique-outcome-455717-k6/locations/us-west1/ragCorpora/4611686018427387904

Use this script to (re)import documents into it -- e.g. after editing files
in sample_documents/, or adding your own real enterprise docs there.

Usage:
    python3 setup_corpus.py                     # import local sample_documents/*.md
    python3 setup_corpus.py --gcs gs://my-bucket/docs/*.md
"""
import argparse
import glob
import os
import sys

import vertexai
from vertexai.preview import rag

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "unique-outcome-455717-k6")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-west1")
CORPUS_RESOURCE_NAME = os.environ.get(
    "RAG_CORPUS_RESOURCE_NAME",
    "projects/unique-outcome-455717-k6/locations/us-west1/ragCorpora/4611686018427387904",
)


def import_local_files(local_glob: str = "sample_documents/*.md"):
    """Uploads local files to a temp GCS location first -- Vertex RAG import
    requires GCS or Drive URIs, it can't ingest local filesystem paths directly."""
    print(
        "NOTE: Vertex AI RAG Engine's import_files requires files to already be in "
        "GCS (or Google Drive) -- it cannot read local disk paths. Upload first, e.g.:\n"
        "  gsutil mb -l "
        f"{LOCATION} gs://YOUR_BUCKET   # if you don't have one yet\n"
        f"  gsutil cp {local_glob} gs://YOUR_BUCKET/docs/\n"
        "then re-run this script with --gcs 'gs://YOUR_BUCKET/docs/*.md'"
    )


def import_gcs_files(gcs_glob: str):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Importing into corpus: {CORPUS_RESOURCE_NAME}")
    response = rag.import_files(
        CORPUS_RESOURCE_NAME,
        [gcs_glob],
        chunk_size=512,
        chunk_overlap=100,
    )
    print("Import result:", response)


def list_corpus_files():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    files = rag.list_files(CORPUS_RESOURCE_NAME)
    for f in files:
        print(f.display_name, f.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs", help="GCS glob to import, e.g. gs://bucket/docs/*.md")
    parser.add_argument("--list", action="store_true", help="List files currently in the corpus")
    args = parser.parse_args()

    if args.list:
        list_corpus_files()
    elif args.gcs:
        import_gcs_files(args.gcs)
    else:
        import_local_files()
