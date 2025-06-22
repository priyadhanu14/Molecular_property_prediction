# 🛡️ CodeGuard AI – AI-Powered Vulnerability Detection with LLMs 🔍

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/priyadhanu14/AI-powered-vulnerability-detection_LLM/ci.yml?label=CI)](../../actions)
[![Issues](https://img.shields.io/github/issues/priyadhanu14/AI-powered-vulnerability-detection_LLM)](../../issues)
[![Stars](https://img.shields.io/github/stars/priyadhanu14/AI-powered-vulnerability-detection_LLM)](../../stargazers)

**CodeGuard AI** is a lightweight, extensible framework that combines **large-language models (LLMs)** with **static-analysis signals** to find, explain, and rank security flaws in source code.  
It couples *retrieval-augmented generation* (RAG) with graph-based context construction to produce **human-readable, line-level vulnerability reports**—turning hours of code review into minutes.

<p align="center">
  <img src="docs/assets/demo_screenshot.png" alt="Demo screenshot showing vulnerability findings" width="780">
</p>

---

## ✨ Key Features
| Category | Highlights |
|----------|------------|
| **Multi-Language Support** | Java • Python • JavaScript (TypeScript & C++ on roadmap) |
| **RAG Pipeline** | Combines AST/CFG graph snippets with vulnerability KBs to ground the LLM’s responses |
| **Explain-as-you-Go** | GPT/Claude (via LangChain) generate step-by-step reasoning + CWE mapping |
| **Risk Scoring** | CVSS-style severity + model confidence fusion |
| **VS Code Extension** | Inline annotations & quick-fix links *(experimental)* |
| **Container-Ready** | One-command `docker compose up` deployment |

---

## 🏗️ High-Level Architecture
```mermaid
flowchart TD
    A[Source Repo] -->|Scan| B(AST/CFG Extractor)
    B --> C{Knowledge Base<br>(CWEs + Docs)}
    C --> D[RAG Context Builder]
    D --> E[LLM (Vulnerability Agent)]
    E --> F[Explanation & Patch Suggester]
    F --> G[Markdown / SARIF Reporter]
