"""
graph_rag.py — GraphRAG for BIMLO Copilot
══════════════════════════════════════════════════════════════════════════════

Two responsibilities:
  1. INGESTION  — extract entities/relationships from document chunks and
                  write them as graph nodes/edges into Neo4j.
  2. RETRIEVAL  — given a user query, decide if it's graph-traversal-suitable,
                  translate it to Cypher, execute it, and return results as
                  RAG-compatible chunk dicts.

Graph schema written by this module:
  (:Entity  {id, name, type, doc_id, filename, description})
  (:Chunk   {id, doc_id, filename, text_preview})
  (:Document{id, filename})

  (:Entity)-[:RELATES_TO {relation, doc_id}]->(:Entity)
  (:Entity)-[:MENTIONED_IN]->(:Chunk)
  (:Chunk)-[:BELONGS_TO]->(:Document)

Entity types extracted (domain-aware for Telecom + BIM):
  SITE, NODE, SWITCH, ROUTER, FIBER_CABLE, PORT, RACK, ANTENNA, BBU, RRH,
  STOREY, ROOM, WALL, BEAM, COLUMN, MATERIAL, ZONE,
  SPEC, STANDARD, PARAMETER, MEASUREMENT, ORGANIZATION, PERSON, DOCUMENT

Usage:
  from graph_rag import GraphRAGEngine
  engine = GraphRAGEngine()                         # call once, singleton inside
  engine.ingest_chunks(doc_id, filename, chunks)    # called after vector store add
  results = engine.query(user_query)                # returns list[dict] like vector chunks
  is_graph = engine.is_graph_query(user_query)      # True if Cypher makes sense

Env vars (inherit from neo4j_auth.py — no duplication needed):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
from typing import List, Dict, Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
# Neo4j connection — reuse the driver from neo4j_auth to avoid double-connect
# ─────────────────────────────────────────────────────────────────────────────

def _get_driver():
    """Reuse the singleton driver from neo4j_auth if loaded, else create own."""
    try:
        import sys
        for mod_name in ("neo4j_auth", "services.neo4j_auth"):
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "get_driver"):
                return mod.get_driver()
        # Not loaded yet — import it
        from neo4j_auth import get_driver
        return get_driver()
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Neo4j: {e}")


def _run_graph(cypher: str, params: dict = None) -> List[Dict]:
    """Run a Cypher query on the graph database."""
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = _get_driver()
    with driver.session(database=db) as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction via LLM
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_SYSTEM = """You are a knowledge graph extraction assistant for telecom and BIM engineering documents.

Extract entities and relationships from the provided text.

Entity types to look for:
- SITE: physical locations, sites, towers, rooftops, pylons
- NODE: network nodes, fiber nodes, distribution nodes
- SWITCH: network switches, core switches, aggregation switches
- ROUTER: routers, PE routers, CE routers
- FIBER_CABLE: fiber cables, optical cables, fiber spans
- PORT: ports, interfaces, connectors
- RACK: equipment racks, cabinets
- ANTENNA: antennas, sectors, panels
- BBU: baseband units
- RRH: remote radio heads, radio units
- STOREY: floors, levels, storeys in a building
- ROOM: rooms, technical rooms, server rooms
- WALL: walls, partitions
- BEAM: structural beams
- COLUMN: structural columns
- MATERIAL: materials (concrete, steel, glass, etc.)
- ZONE: zones, areas, coverage zones
- SPEC: specifications, standards, norms
- PARAMETER: technical parameters, measurements, values
- ORGANIZATION: companies, operators, vendors
- PERSON: engineers, project managers, contacts

Return ONLY a valid JSON object with this exact structure:
{
  "entities": [
    {"name": "Switch A", "type": "SWITCH", "description": "Core aggregation switch"},
    {"name": "Node 12", "type": "NODE", "description": "Distribution fiber node"}
  ],
  "relationships": [
    {"source": "Node 12", "relation": "CONNECTS_TO", "target": "Switch A", "description": "via 48-fiber cable"}
  ]
}

If no clear entities are found, return {"entities": [], "relationships": []}.
Do NOT include any text outside the JSON."""


def _extract_entities_from_chunk(text: str, filename: str) -> Dict:
    """
    Call the LLM to extract entities and relationships from one chunk.
    Returns {"entities": [...], "relationships": [...]} or empty on failure.
    """
    from llm_client import call_llm

    # Only extract from chunks with enough content
    if len(text.strip()) < 100:
        return {"entities": [], "relationships": []}

    prompt = f"Document: {filename}\n\nText:\n{text[:1500]}"

    try:
        raw = call_llm(
            prompt=prompt,
            system_prompt=_ENTITY_SYSTEM,
            max_tokens=800,
            temperature=0.0,
            task="classify",
        )
        if not raw:
            return {"entities": [], "relationships": []}

        # Parse JSON — strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        return parsed

    except (json.JSONDecodeError, Exception) as e:
        # Non-fatal — skip this chunk
        return {"entities": [], "relationships": []}


# ─────────────────────────────────────────────────────────────────────────────
# Graph schema setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_graph_schema():
    """Create constraints and indexes for the graph schema."""
    statements = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_gid IF NOT EXISTS FOR (c:GraphChunk) REQUIRE c.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    for stmt in statements:
        try:
            _run_graph(stmt)
        except Exception:
            pass  # may already exist
    print("✅ graph_rag: schema ready")


# ─────────────────────────────────────────────────────────────────────────────
# Cypher query detection
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that strongly suggest a graph traversal question
_GRAPH_TRIGGERS = [
    r"\bconnect(ed|s|ion)?\b",
    r"\blink(ed|s)?\b",
    r"\btopolog(y|ies)\b",
    r"\bpath\b",
    r"\broute?\b",
    r"\bneighbo(u)?r\b",
    r"\brelat(ed|ion|ions|ionship)?\b",
    r"\bdepend(s|ency|encies)?\b",
    r"\bbetween\b.{0,30}\band\b",
    r"\bwhich .{0,20}(connect|link|attach|belong|feed)\b",
    r"\bwhat (is|are) connected\b",
    r"\bflows?\b",
    r"\bspans?\b",
    r"\binterface\b",
    r"\buplift\b",
    r"\bfeeds?\b",
    r"\bserve[sd]?\b",
    # BIM spatial
    r"\bon (floor|level|storey)\b",
    r"\bin (room|zone|area)\b",
    r"\bcontain(s|ed)?\b",
    r"\badjacent\b",
    r"\bsupport(ed|s)?\b",
]

_GRAPH_TRIGGER_RE = re.compile(
    "|".join(_GRAPH_TRIGGERS), re.IGNORECASE
)


def is_graph_query(query: str) -> bool:
    """Return True if the query is likely asking about relationships/topology."""
    return bool(_GRAPH_TRIGGER_RE.search(query))


# ─────────────────────────────────────────────────────────────────────────────
# Cypher generation
# ─────────────────────────────────────────────────────────────────────────────

_CYPHER_SYSTEM = """You are a Neo4j Cypher expert for a telecom and BIM knowledge graph.

Graph schema:
  (:Entity {id, name, type, doc_id, filename, description})
  (:GraphChunk {id, doc_id, filename, text_preview})
  (:Entity)-[:RELATES_TO {relation, description}]->(:Entity)
  (:Entity)-[:MENTIONED_IN]->(:GraphChunk)

Entity types: SITE, NODE, SWITCH, ROUTER, FIBER_CABLE, PORT, RACK, ANTENNA, BBU, RRH,
              STOREY, ROOM, WALL, BEAM, COLUMN, MATERIAL, ZONE, SPEC, PARAMETER,
              ORGANIZATION, PERSON

Given the user's natural language question, generate a Cypher READ query.
Rules:
- Use MATCH, OPTIONAL MATCH, WHERE, RETURN only — NO CREATE/MERGE/DELETE
- Use case-insensitive matching: WHERE toLower(e.name) CONTAINS toLower($term)
- Always LIMIT results to 20 max
- Return human-readable fields: entity names, types, relations, descriptions
- If the query asks about connections, traverse [:RELATES_TO] edges
- If you cannot generate a safe read-only query, return exactly: SKIP

Return ONLY the Cypher query or "SKIP", nothing else."""


def _generate_cypher(query: str) -> Optional[str]:
    """Ask the LLM to generate a Cypher query for this natural language question."""
    from llm_client import call_llm

    # Extract potential entity names from the query to use as parameters
    prompt = f'User question: "{query}"\n\nGenerate the Cypher query:'

    try:
        raw = call_llm(
            prompt=prompt,
            system_prompt=_CYPHER_SYSTEM,
            max_tokens=300,
            temperature=0.0,
            task="classify",
        )
        if not raw or raw.strip().upper() == "SKIP":
            return None

        # Strip markdown fences
        cypher = re.sub(r"```(?:cypher)?|```", "", raw).strip()

        # Safety check — only allow read operations
        dangerous = re.compile(
            r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+db\.)\b",
            re.IGNORECASE,
        )
        if dangerous.search(cypher):
            print(f"⚠️  graph_rag: blocked unsafe Cypher: {cypher[:100]}")
            return None

        return cypher

    except Exception as e:
        print(f"⚠️  graph_rag: Cypher generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main engine class
# ─────────────────────────────────────────────────────────────────────────────

class GraphRAGEngine:
    """
    GraphRAG engine — entity ingestion + graph-aware retrieval.

    Designed to run alongside (not replace) the vector store.
    The RAG engine calls it opportunistically:
      - After upload: ingest_chunks() builds the knowledge graph
      - At query time: query() returns graph-sourced context chunks
    """

    def __init__(self):
        self._available = False
        try:
            _run_graph("RETURN 1 AS ping")
            _setup_graph_schema()
            self._available = True
            print("✅ graph_rag: GraphRAG engine ready")
        except Exception as e:
            print(f"⚠️  graph_rag: Neo4j not reachable — GraphRAG disabled ({e})")

    @property
    def available(self) -> bool:
        return self._available

    # ── INGESTION ────────────────────────────────────────────────────────────

    def ingest_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: List[Dict],
        max_chunks: int = 30,
    ) -> Dict:
        """
        Extract entities from document chunks and write them to Neo4j.

        We sample up to `max_chunks` chunks (every Nth) to avoid hammering
        the LLM on large documents. Returns extraction stats.

        Called from main.py after vector_store.add_document().
        Non-fatal — never raises, always returns stats dict.
        """
        if not self._available:
            return {"entities": 0, "relationships": 0, "skipped": True}

        total_entities = 0
        total_relations = 0
        errors = 0

        # Sample chunks evenly across the document
        step = max(1, len(chunks) // max_chunks)
        sampled = chunks[::step][:max_chunks]

        print(f"🕸️  graph_rag: extracting from {len(sampled)}/{len(chunks)} chunks of '{filename}'…")
        t0 = time.time()

        for i, chunk in enumerate(sampled):
            text = chunk.get("text", "")
            chunk_id = f"{doc_id}_chunk_{i}"

            try:
                # Write the chunk node
                _run_graph(
                    """
                    MERGE (gc:GraphChunk {id: $chunk_id})
                    SET gc.doc_id       = $doc_id,
                        gc.filename     = $filename,
                        gc.text_preview = $preview
                    """,
                    {
                        "chunk_id": chunk_id,
                        "doc_id":   doc_id,
                        "filename": filename,
                        "preview":  text[:300],
                    },
                )

                # Extract entities via LLM
                extracted = _extract_entities_from_chunk(text, filename)
                entities      = extracted.get("entities", [])
                relationships = extracted.get("relationships", [])

                # Write entity nodes
                entity_ids: Dict[str, str] = {}  # name → node id
                for ent in entities:
                    name = ent.get("name", "").strip()
                    if not name:
                        continue
                    ent_id = f"{doc_id}_{name.lower().replace(' ', '_')}"
                    entity_ids[name] = ent_id

                    _run_graph(
                        """
                        MERGE (e:Entity {id: $ent_id})
                        ON CREATE SET
                            e.name        = $name,
                            e.type        = $type,
                            e.doc_id      = $doc_id,
                            e.filename    = $filename,
                            e.description = $desc
                        ON MATCH SET
                            e.description = CASE
                                WHEN e.description IS NULL OR e.description = ''
                                THEN $desc ELSE e.description END
                        WITH e
                        MATCH (gc:GraphChunk {id: $chunk_id})
                        MERGE (e)-[:MENTIONED_IN]->(gc)
                        """,
                        {
                            "ent_id":   ent_id,
                            "name":     name,
                            "type":     ent.get("type", "UNKNOWN"),
                            "doc_id":   doc_id,
                            "filename": filename,
                            "desc":     ent.get("description", ""),
                            "chunk_id": chunk_id,
                        },
                    )
                    total_entities += 1

                # Write relationships
                for rel in relationships:
                    src_name  = rel.get("source", "").strip()
                    tgt_name  = rel.get("target", "").strip()
                    relation  = rel.get("relation", "RELATES_TO").upper().replace(" ", "_")
                    desc      = rel.get("description", "")

                    if not src_name or not tgt_name:
                        continue

                    src_id = entity_ids.get(src_name, f"{doc_id}_{src_name.lower().replace(' ', '_')}")
                    tgt_id = entity_ids.get(tgt_name, f"{doc_id}_{tgt_name.lower().replace(' ', '_')}")

                    _run_graph(
                        """
                        MERGE (src:Entity {id: $src_id})
                        ON CREATE SET src.name = $src_name, src.doc_id = $doc_id, src.filename = $filename
                        MERGE (tgt:Entity {id: $tgt_id})
                        ON CREATE SET tgt.name = $tgt_name, tgt.doc_id = $doc_id, tgt.filename = $filename
                        MERGE (src)-[r:RELATES_TO {relation: $relation, doc_id: $doc_id}]->(tgt)
                        SET r.description = $desc
                        """,
                        {
                            "src_id":   src_id,
                            "src_name": src_name,
                            "tgt_id":   tgt_id,
                            "tgt_name": tgt_name,
                            "relation": relation,
                            "doc_id":   doc_id,
                            "filename": filename,
                            "desc":     desc,
                        },
                    )
                    total_relations += 1

            except Exception as e:
                errors += 1
                if errors <= 3:  # don't spam logs
                    print(f"⚠️  graph_rag: chunk {i} ingestion error: {e}")

        elapsed = time.time() - t0
        print(
            f"✅ graph_rag: ingested '{filename}' — "
            f"{total_entities} entities, {total_relations} relationships "
            f"({elapsed:.1f}s, {errors} errors)"
        )
        return {
            "entities":      total_entities,
            "relationships": total_relations,
            "errors":        errors,
            "elapsed":       elapsed,
        }

    # ── RETRIEVAL ────────────────────────────────────────────────────────────

    def query(self, user_query: str, doc_id: Optional[str] = None) -> List[Dict]:
        """
        Run a graph-aware query against Neo4j.

        Returns a list of chunk-like dicts compatible with the RAG engine's
        retrieved_chunks format so they can be merged with vector results.

        Returns [] if graph is unavailable or no results found.
        """
        if not self._available:
            return []

        print(f"🕸️  graph_rag: running graph query for '{user_query[:80]}'")
        t0 = time.time()

        # Step 1: Generate Cypher
        cypher = _generate_cypher(user_query)
        if not cypher:
            # Fallback: keyword-based entity lookup
            cypher = self._keyword_fallback_cypher(user_query)

        print(f"   Cypher: {cypher[:120]}…" if len(cypher) > 120 else f"   Cypher: {cypher}")

        # Step 2: Execute
        try:
            rows = _run_graph(cypher)
        except Exception as e:
            print(f"⚠️  graph_rag: Cypher execution failed ({e}), trying fallback")
            try:
                rows = _run_graph(self._keyword_fallback_cypher(user_query))
            except Exception:
                return []

        if not rows:
            print(f"   graph_rag: no results ({time.time()-t0:.2f}s)")
            return []

        # Step 3: Format as RAG chunks
        chunks = self._rows_to_chunks(rows, user_query)
        print(f"   graph_rag: {len(chunks)} graph chunks ({time.time()-t0:.2f}s)")
        return chunks

    def _keyword_fallback_cypher(self, query: str) -> str:
        """
        Simple keyword-based entity lookup when LLM Cypher generation fails.
        Extracts meaningful words and searches entity names.
        """
        # Pull out likely entity name words (capitalized, 3+ chars)
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]{2,}\b', query)
        stop  = {"what", "which", "where", "when", "how", "the", "are",
                 "is", "connected", "connect", "between", "and", "or",
                 "from", "to", "of", "in", "on", "at", "for"}
        terms = [w for w in words if w.lower() not in stop][:3]

        if not terms:
            return "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.description AS description LIMIT 10"

        conditions = " OR ".join(
            f"toLower(e.name) CONTAINS toLower('{t}')" for t in terms
        )
        return f"""
        MATCH (e:Entity)
        WHERE {conditions}
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)
        RETURN e.name AS name, e.type AS type, e.description AS description,
               r.relation AS relation, e2.name AS related_to, e2.type AS related_type,
               r.description AS relation_desc, e.filename AS filename
        LIMIT 20
        """

    def _rows_to_chunks(self, rows: List[Dict], query: str) -> List[Dict]:
        """
        Convert Neo4j result rows into RAG-compatible chunk dicts.
        Groups by source entity to produce readable context paragraphs.
        """
        if not rows:
            return []

        # Group relationships by source entity
        entity_map: Dict[str, Dict] = {}
        for row in rows:
            name = row.get("name") or row.get("source_name") or ""
            if not name:
                continue

            if name not in entity_map:
                entity_map[name] = {
                    "type":        row.get("type", "ENTITY"),
                    "description": row.get("description", ""),
                    "filename":    row.get("filename", "knowledge graph"),
                    "relations":   [],
                }

            rel = row.get("relation") or row.get("relationship")
            tgt = row.get("related_to") or row.get("target_name")
            if rel and tgt:
                entity_map[name]["relations"].append({
                    "relation":    rel,
                    "target":      tgt,
                    "target_type": row.get("related_type", ""),
                    "desc":        row.get("relation_desc", ""),
                })

        # Build one chunk per entity with all its relationships
        chunks = []
        for entity_name, info in entity_map.items():
            lines = [
                f"Entity: {entity_name} [{info['type']}]",
            ]
            if info["description"]:
                lines.append(f"Description: {info['description']}")

            if info["relations"]:
                lines.append("Relationships:")
                for rel in info["relations"][:10]:
                    rel_line = f"  - {rel['relation']} → {rel['target']} [{rel['target_type']}]"
                    if rel["desc"]:
                        rel_line += f" ({rel['desc']})"
                    lines.append(rel_line)

            text = "\n".join(lines)

            chunks.append({
                "text":         text,
                "distance":     0.05,  # treat graph results as high-confidence
                "rerank_score": 2.0,   # always surface above vector results
                "id":           f"graph_{entity_name}",
                "metadata": {
                    "filename":    info["filename"],
                    "doc_type":    "graph",
                    "source":      "knowledge_graph",
                    "has_images":  False,
                    "has_tables":  False,
                    "entity_name": entity_name,
                    "entity_type": info["type"],
                },
            })

        return chunks

    # ── STATS ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return entity/relationship counts for the health endpoint."""
        if not self._available:
            return {"available": False}
        try:
            entity_count = _run_graph("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]
            rel_count    = _run_graph("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n")[0]["n"]
            doc_count    = _run_graph("MATCH (gc:GraphChunk) RETURN count(DISTINCT gc.doc_id) AS n")[0]["n"]
            return {
                "available":     True,
                "entities":      entity_count,
                "relationships": rel_count,
                "documents":     doc_count,
            }
        except Exception as e:
            return {"available": True, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import and use directly
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[GraphRAGEngine] = None


def get_engine() -> GraphRAGEngine:
    global _engine
    if _engine is None:
        _engine = GraphRAGEngine()
    return _engine
