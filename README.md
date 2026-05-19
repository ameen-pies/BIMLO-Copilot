```
░████████   ░██████░███     ░███ ░██           ░██████        ░██████                        ░██░██               ░██    
░██    ░██    ░██  ░████   ░████ ░██          ░██   ░██      ░██   ░██                          ░██               ░██    
░██    ░██    ░██  ░██░██ ░██░██ ░██         ░██     ░██    ░██         ░███████  ░████████  ░██░██  ░███████  ░████████ 
░████████     ░██  ░██ ░████ ░██ ░██         ░██     ░██    ░██        ░██    ░██ ░██    ░██ ░██░██ ░██    ░██    ░██    
░██     ░██   ░██  ░██  ░██  ░██ ░██         ░██     ░██    ░██        ░██    ░██ ░██    ░██ ░██░██ ░██    ░██    ░██    
░██     ░██   ░██  ░██       ░██ ░██          ░██   ░██      ░██   ░██ ░██    ░██ ░███   ░██ ░██░██ ░██    ░██    ░██    
░█████████  ░██████░██       ░██ ░██████████   ░██████        ░██████   ░███████  ░██░█████  ░██░██  ░███████      ░████ 
                                                                                  ░██                                    
                                                                                  ░██                                    
                                                                                                                         
```

<h3 align="center">AI Copilot for Telecom & BIM Workflows</h3>

<p align="center">
  Chat over documents · Generate reports · Track industry news · Voice interactions · CAD/IFC context
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/TypeScript-5-blue?logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/LangChain-Latest-omidou?logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/Neo4j-5-018BFF?logo=neo4j" alt="Neo4j">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/i18n-EN%20%7C%20FR%20%7C%20AR-green" alt="i18n">
</p>

---

## Why This Exists

Teams dealing with dense technical information don't have time to dig through PDFs, project docs, knowledge graphs, and market updates by hand.

BIMLO pulls together document-aware RAG, graph-backed memory, report generation, news intelligence, voice interfaces, and CAD/IFC-aware flows into one workspace. Upload material, ask hard questions, keep session context, get answers with sources.

## Features

| Surface | What It Does |
|---|---|
| **Chat** (`/chat`) | RAG over uploaded documents with per-session memory, citations, confidence scores |
| **News** (`/news`) | Industry news intelligence with dedicated chat agent |
| **Call** (`/call`) | Voice transcription and call-oriented flows |
| **Admin** (`/admin`) | System health monitoring, session stats, provider status |
| **Reports** | Structured report generation from retrieved context |
| **CAD/IFC** | BIM-aware assistant through dedicated agent services |

**Backend capabilities:**

- Multi-provider LLM routing (Cloudflare Workers, Groq, OpenRouter) with automatic fallback
- 35+ prompt templates for specialized agents (RAG, reports, news, graph, vision, voice)
- LangGraph-powered document ingestion pipeline
- ChromaDB vector retrieval with reranking
- Neo4j auth, session persistence, and graph memory
- LLM judge for answer quality evaluation
- Autocomplete and suggestion flows
- Image/PDF vision support in RAG answers

## Stack

| Layer | Tech |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, TanStack Query, Radix UI, Framer Motion |
| **Backend** | FastAPI, LangChain + LangGraph, ChromaDB, Sentence Transformers, Neo4j |
| **LLM Providers** | Cloudflare Workers (Llama 3.3 70B), Groq, OpenRouter |
| **Infra** | Docker Compose, Doppler (secrets), observability hooks |

## Architecture

```
Frontend (React/Vite :5173)
        │
        ▼
   FastAPI API (:8000)
        │
        ├─► Auth + Sessions ────────► Neo4j (:7687)
        ├─► Document Ingestion ─────► LangGraph pipeline
        ├─► Vector Retrieval ───────► ChromaDB (:8001)
        ├─► RAG / Routing ──────────► Multi-provider LLM
        ├─► Report Agent ───────────► Structured output
        ├─► News Pipeline ──────────► News chat agent
        ├─► Graph Agent ────────────► Data extraction + charts
        ├─► Voice / Call ───────────► Transcription + rewrite
        └─► CAD/IFC Agent ──────────► BIM context bridge
```

## Project Structure

```
BIMLO/
├── Backend/
│   ├── routers/          API endpoints (chat, documents, news, reports, sessions, providers, health)
│   ├── services/         Core logic (RAG engine, ingestion, agents, vector store, news, voice, CAD)
│   ├── prompts/          35+ LLM prompt templates
│   ├── models/           Data models
│   ├── core/             Config and shared utilities
│   ├── evals/            Evaluation harness
│   ├── providers.json    Multi-provider LLM config
│   ├── main.py           FastAPI entry point
│   └── docker-compose.yml
├── Frontend/
│   ├── src/
│   │   ├── pages/        Chat, News, Call, Admin, Landing
│   │   ├── components/   UI system (Radix primitives, chat, effects)
│   │   ├── services/     API client
│   │   ├── locales/      en.json, fr.json, ar.json
│   │   ├── hooks/        Custom React hooks
│   │   └── context/      React context providers
│   └── package.json
└── data/                 Runtime data, uploads, generated artifacts
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- Docker + Docker Compose
- API key for at least one LLM provider

### Environment Setup

Create `Backend/.env` with:

```env
# LLM providers (at least one)
GROQ_API_KEY=...
CF_API_KEY=...

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Storage
DATA_DIR=./data

# Frontend
VITE_API_URL=http://localhost:8000
```

### Run with Docker (recommended)

```bash
cd Backend
docker compose up --build
```

Spins up: Neo4j, ChromaDB, and the FastAPI backend.

Then start the frontend separately:

```bash
cd Frontend
npm install
npm run dev
```

### Run without Docker

```bash
# Backend
cd Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd Frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| Backend API | `http://localhost:8000` |
| Neo4j Browser | `http://localhost:7474` |
| ChromaDB | `http://localhost:8001` |

## Internationalization

Full i18n support with per-message RTL detection:

- **English** (en)
- **French** (fr)
- **Arabic** (ar) — full RTL layout support

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload documents for RAG ingestion |
| `/documents` | GET | List uploaded documents |
| `/documents/{id}` | DELETE | Remove a document |
| `/query` | POST | Ask a question over documents |
| `/generate-report` | POST | Generate structured report |
| `/health` | GET | System health check |
| `/news-conversations` | GET | News chat sessions |
| `/sessions` | GET | User sessions |
| `/providers` | GET | Available LLM providers |

Additional routes for auth, suggestions, voice, autocomplete, and CAD/IFC flows.

## Developer Notes

- Directory names are capitalized: `Backend/`, `Frontend/`
- Secrets management: `.env` for dev, Doppler for production (`docker compose --profile doppler up`)
- LLM provider config lives in `Backend/providers.json` — add/remove providers without code changes
- Prompt templates in `Backend/prompts/` — 35+ specialized prompts for different agents
- Never commit credentials, API keys, or `.env` files

## License

Proprietary. All rights reserved.
