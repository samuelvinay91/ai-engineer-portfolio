#!/usr/bin/env bash
# =============================================================================
# Create separate GitHub repos for each AI Engineer Portfolio project
# =============================================================================
# Usage:
#   ./create-github-repos.sh                    # Create all repos
#   ./create-github-repos.sh llm-playground     # Create one repo
#   DRY_RUN=1 ./create-github-repos.sh          # Preview without creating
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STANDALONE_DIR="${SCRIPT_DIR}/standalone-repos"
GITHUB_USER="${GITHUB_USER:-$(gh api user -q .login 2>/dev/null || echo "YOUR_USERNAME")}"
DRY_RUN="${DRY_RUN:-0}"

# Projects
PROJECTS=(
  "llm-playground"
  "customer-support-chatbot"
  "ask-the-web-agent"
  "deep-research"
  "image-generation"
  "capstone-multiagent"
  "agent-rag"
  "mcp-a2a"
)

# Topics for each repo
declare -A TOPICS=(
  ["llm-playground"]="llm tokenization bpe transformer generation-strategies fastapi docker cloud-native ai-engineer"
  ["customer-support-chatbot"]="chatbot customer-support lora peft fine-tuning prompt-engineering langchain fastapi docker ai-engineer"
  ["ask-the-web-agent"]="langgraph agent perplexity web-search rag langchain fastapi docker cloud-native ai-engineer"
  ["deep-research"]="deep-research chain-of-thought tree-of-thought reasoning langgraph inference-scaling fastapi docker ai-engineer"
  ["image-generation"]="image-generation diffusion dall-e stable-diffusion flux text-to-image fastapi docker ai-engineer"
  ["capstone-multiagent"]="multi-agent supervisor-pattern orchestration langgraph langchain fastapi docker cloud-native ai-engineer"
  ["agent-rag"]="rag retrieval-augmented-generation vector-search qdrant langgraph langchain fastapi docker ai-engineer"
  ["mcp-a2a"]="mcp model-context-protocol a2a agent-to-agent agentcard interoperability fastapi docker ai-engineer"
)

# Descriptions
declare -A DESCRIPTIONS=(
  ["llm-playground"]="Interactive LLM Playground - Explore tokenization, text generation strategies, and transformer architectures"
  ["customer-support-chatbot"]="Production customer support chatbot with PEFT/LoRA fine-tuning and advanced prompt engineering"
  ["ask-the-web-agent"]="Perplexity-like Ask-the-Web agent with search, synthesis, and citation - demonstrating agentic patterns"
  ["deep-research"]="Deep Research capability with reasoning models, CoT prompting, and inference-time scaling"
  ["image-generation"]="Cloud-native Image Generation Service with diffusion models, VAE, and multi-provider support"
  ["capstone-multiagent"]="Capstone: Multi-Agent AI Platform orchestrating specialized agents for complex task solving"
  ["agent-rag"]="Advanced Agent & RAG System with hierarchical retrieval, query decomposition, and multi-agent patterns"
  ["mcp-a2a"]="MCP Server/Client implementation and A2A protocol with AgentCards for agent interoperability"
)

create_repo() {
  local name="$1"
  local desc="${DESCRIPTIONS[$name]}"
  local topics="${TOPICS[$name]}"
  local repo_dir="${STANDALONE_DIR}/${name}"

  if [[ ! -d "$repo_dir" ]]; then
    echo "ERROR: Directory $repo_dir not found. Run generate_standalone_repos.py first."
    return 1
  fi

  echo ""
  echo "================================================================"
  echo "  Creating repo: ${GITHUB_USER}/${name}"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY RUN] Would create: ${GITHUB_USER}/${name}"
    echo "  [DRY RUN] Description: ${desc}"
    echo "  [DRY RUN] Topics: ${topics}"
    return 0
  fi

  cd "$repo_dir"

  # Initialize git repo
  git init -b main
  git add .
  git commit -m "Initial commit: ${name} - AI Engineer Portfolio Project"

  # Create GitHub repo
  gh repo create "${name}" \
    --public \
    --description "${desc}" \
    --source . \
    --remote origin \
    --push

  # Set topics
  for topic in $topics; do
    gh repo edit "${GITHUB_USER}/${name}" --add-topic "$topic" 2>/dev/null || true
  done

  echo "  Created: https://github.com/${GITHUB_USER}/${name}"
  cd - > /dev/null
}

# Main
echo "GitHub User: ${GITHUB_USER}"
echo "Standalone Dir: ${STANDALONE_DIR}"
echo ""

if [[ $# -gt 0 ]]; then
  # Create specific repo
  create_repo "$1"
else
  # Create all repos
  for name in "${PROJECTS[@]}"; do
    create_repo "$name"
  done
fi

echo ""
echo "Done! All repos created at https://github.com/${GITHUB_USER}"
