#!/usr/bin/env bash
# Deploy root_agent to Vertex AI Agent Engine.
#
# Verified against the installed ADK version's actual `adk deploy agent_engine`
# CLI (--staging_bucket is deprecated/unused in this version; there is no
# --set_env_vars flag -- env vars are instead read from the project's .env
# file and passed through as the deployed agent's runtime env_vars config,
# NOT shipped as a raw file). Re-check `adk deploy agent_engine --help`
# yourself if you upgrade ADK later, since this surface changes across versions.
#
# Before running:
#   1. gcloud auth application-default login
#   2. gcloud config set project unique-outcome-455717-k6
#   3. Make sure this project's .env has your real values.
#   4. Make sure .gitignore excludes your venv folder (.venv/, venv/, etc.)
#      -- adk deploy bundles everything NOT in .gitignore, and a venv folder
#      will blow past the 8MB deploy payload limit.

set -euo pipefail

PROJECT_ID="unique-outcome-455717-k6"
REGION="us-west1"

# This script lives inside the project folder itself (next to agent.py).
# adk deploy needs to be invoked from the PARENT directory, with the project
# folder name as its argument -- so cd up one level regardless of where you
# launched this script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_FOLDER_NAME="$(basename "${SCRIPT_DIR}")"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Deploying '${PROJECT_FOLDER_NAME}' (from ${PARENT_DIR}) to project ${PROJECT_ID} (${REGION})..."
cd "${PARENT_DIR}"

adk deploy agent_engine \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display_name="enterprise-genai-strategist" \
  "${PROJECT_FOLDER_NAME}"

echo ""
echo "---"
echo "adk's own exit code is not reliable for detecting failure in this ADK version"
echo "(it can print 'Deploy failed: ...' and still return 0). Check the output above"
echo "for 'Deploy failed' or 'INVALID_ARGUMENT' before assuming this worked."
echo "On success, check the Agent Engine console: project ${PROJECT_ID}, region ${REGION}."
