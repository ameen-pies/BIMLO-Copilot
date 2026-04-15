# BIMLO

> An AI copilot for telecom and BIM workflows that can chat over documents, generate reports, track industry news, handle voice interactions, and plug into CAD/IFC context.

## Why This Exists

BIMLO is built for teams that deal with dense technical information and do not have time to dig through PDFs, project docs, knowledge graphs, and market updates by hand.

Instead of shipping a basic chatbot, this project pulls together:

- document-aware RAG
- graph-backed memory and auth
- report generation
- news intelligence
- voice interfaces
- CAD/IFC-aware assistant flows

The result is a workspace where users can upload material, ask hard questions, keep session context, and get answers with sources instead of vibes.

## What It Can Do

- Chat with uploaded project documents using retrieval-augmented generation
- Keep per-session memory so conversations stay coherent across turns
- Generate reports from retrieved context
- Surface citations and confidence with answers
- Run telecom/news-specific flows through dedicated agents
- Support voice transcription and call-oriented routes
- Handle authentication and conversation persistence with Neo4j
- Extend into CAD/IFC workflows through dedicated backend services

## Stack

**Frontend**

- React 18
- TypeScript
- Vite
- Tailwind CSS
- TanStack Query
- Radix UI
- Framer Motion / GSAP / 3D visual libraries

**Backend**

- FastAPI
- LangChain + LangGraph
- ChromaDB
- Sentence Transformers
- Neo4j
- Groq-backed LLM calls

## Architecture

```text
Frontend (React/Vite)
    |
    v
FastAPI API layer
    |
    +--> Auth + session persistence (Neo4j)
    +--> Document ingestion pipeline (LangGraph)
    +--> Vector retrieval (ChromaDB)
    +--> RAG / routing / report / news / voice / CAD agents
    |
    v
Answers, sources, reports, session memory
```

## Project Layout

```text
BIMLO/
|- Backend/        FastAPI app, agents, ingestion, auth, retrieval, reports
|- Frontend/       React app, routes, UI system, API client
|- data/           Runtime data, uploads, generated artifacts
`- docker-compose.yml
```

## Main Product Surfaces

- `/` landing page
- `/chat` document-aware copilot experience
- `/news` news intelligence interface
- `/call` voice/call experience

Backend services include document processing, autocomplete, suggestion flows, voice routes, report generation, news pipelines, and CAD/IFC agent support.

## Local Setup

### 1. Prerequisites

- Node.js 18+
- Python 3.10+
- Neo4j instance
- API credentials for the LLM providers you plan to use

### 2. Configure environment variables

Create your local env files and provide values for the variables used by the app.

Common variables referenced in this repo:

- `GROQ_API_KEY`
- `CF_API_KEY`
- `DATA_DIR`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `VITE_API_URL`

Keep secrets in local env files and out of version control.

### 3. Start the backend

```bash
cd Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the frontend

```bash
cd Frontend
npm install
npm run dev
```

Frontend default dev URL:

- `http://localhost:5173`

Backend default dev URL:

- `http://localhost:8000`

## Docker

The repo includes a `docker-compose.yml` with:

- backend service
- frontend development profile
- frontend production profile

Typical usage:

```bash
docker compose --profile dev up --build
```

If you use Docker on a case-sensitive environment, make sure the service build paths match your actual folder names.

## API Notes

Representative backend capabilities include:

- `POST /upload`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `POST /query`
- `POST /generate-report`
- `GET /health`

Additional routes are mounted for auth, suggestions, voice, reports, news chat, and CAD/IFC flows.

## Frontend Highlights

The frontend is more than a dashboard shell. It includes:

- route-based product areas for chat, news, and calls
- a custom visual layer with motion and interactive effects
- auth-aware API access
- reusable UI primitives under `src/components/ui`

## Backend Highlights

The backend centers around a modular services layer:

- `document_processor.py` for ingestion prep
- `vector_store.py` for retrieval storage
- `rag_engine.py` for answer generation
- `ingestion_graph.py` for pipeline orchestration
- `report_agent.py` for structured report output
- `news_pipeline.py` and `news_chat_agent.py` for news workflows
- `cad_ifc_agent.py` for CAD/IFC-specific capabilities
- `neo4j_auth.py` for auth, tokens, and persisted conversation state

## Current Positioning

BIMLO already reads like a serious applied AI product:

- multi-surface frontend
- agent-style backend
- persistent graph memory
- retrieval and reporting
- domain-specific expansion points

That makes it a strong base for internal ops tooling, technical knowledge copilots, or client-facing expert assistants.

## Developer Notes

- The root project uses `Backend/` and `Frontend/` directory names with capital letters.
- The checked-in frontend README started as a generated placeholder and has been replaced with project-specific docs.
- There are active env-driven integrations here, so be careful not to leak credentials in commits, screenshots, or issue threads.



