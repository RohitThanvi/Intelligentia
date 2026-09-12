# Deploy root_agent to Vertex AI Agent Engine (Windows PowerShell version of deploy.sh).
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
#   3. Make sure enterprise_strategist\.env (or Intelligentia\.env, whatever
#      you've named the project folder) has your real values.
#
# Run from PowerShell:  .\deploy.ps1
# If you get an "execution of scripts is disabled" error, run once first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

$ProjectId = "unique-outcome-455717-k6"
$Region    = "us-west1"

# This script lives inside the project folder itself (next to agent.py).
# adk deploy needs to be invoked from the PARENT directory, with the project
# folder name as its argument -- so we cd up one level regardless of where
# you launched this script from.
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectFolderName = Split-Path -Leaf $ProjectDir
$ParentDir = Split-Path -Parent $ProjectDir

Write-Host "Deploying '$ProjectFolderName' (from $ParentDir) to project $ProjectId ($Region)..."

Push-Location $ParentDir
try {
    adk deploy agent_engine `
      --project=$ProjectId `
      --region=$Region `
      --display_name="enterprise-genai-strategist" `
      $ProjectFolderName
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "---"
Write-Host "adk's own exit code is not reliable for detecting failure in this ADK version"
Write-Host "(it can print 'Deploy failed: ...' and still return 0). Check the output above"
Write-Host "for 'Deploy failed' or 'INVALID_ARGUMENT' before assuming this worked."
Write-Host "On success, check the Agent Engine console: project $ProjectId, region $Region."
