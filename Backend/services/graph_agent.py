"""
graph_agent.py — Intelligent Chart Generation Agent

Detects when a user is asking for a graph/chart, extracts structured data
from retrieved document chunks, decides on the best chart type, and returns
a Chart.js-compatible config that the frontend can render directly.

How it plugs in:
  1. Router in rag_engine.py picks route = "graph" for chart requests
  2. RAGEngine._build_graph() adds a graph_node
  3. graph_node calls GraphAgent.build_chart()
  4. The result is stored in state["analytics"] with type="chart_config"
     so the frontend knows to render a Chart.js canvas instead of markdown

Mount pattern (in rag_engine.py):
    from graph_agent import GraphAgent, is_graph_request

Add to AgentState:
    route: Optional[Literal["direct", "rag", "iterative_rag", "analytics",
                            "transform", "define", "graph"]]  # ← add "graph"

Add to router prompt:
    - graph: the user is asking for a chart, graph, plot, or visual
      representation of data from the documents — bar charts, line graphs,
      pie charts, scatter plots, timelines, etc.

Add to _build_graph():
    workflow.add_node("graph_node", _wrap("graph_node", self.graph_node))
    # In router edges:
    "graph": "retrieve",
    # In check_retrieval:
    if s["route"] == "graph": return "graph"
    # New edge:
    workflow.add_edge("graph_node", END)

Add to RAGEngine.__init__():
    from graph_agent import GraphAgent
    self.graph_agent = GraphAgent(
        api_key=os.getenv("CF_API_KEY", ""),
        base_url=os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev"),
    )

Add graph_node method to RAGEngine (see bottom of this file for the method body).

Env vars (shared with rest of engine):
    CF_API_KEY  — required
    CF_API_URL  — optional (default: Cloudflare worker URL)
"""

from __future__ import annotations

import os
import re
import json
import time
import requests
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# CHART TYPE CATALOGUE
# Maps intent keywords → Chart.js type + recommended structure
# ────────────────────────────────────────────────────────────────────────────

_CHART_HINTS: List[Tuple[List[str], str]] = [
    (["pie", "donut", "proportion", "share", "breakdown", "distribution"], "pie"),
    (["line", "trend", "over time", "timeline", "progress", "evolution",
      "growth", "month", "year", "week", "daily", "quarterly"], "line"),
    (["scatter", "correlation", "relationship between", "plot"], "scatter"),
    (["area", "cumulative", "stacked"], "line"),   # Chart.js uses fill:true
    (["bar", "compare", "comparison", "ranking", "top", "versus",
      "vs", "highest", "lowest", "most", "least", "count",
      "histogram", "frequency"], "bar"),
    (["horizontal", "ranked"], "bar"),
]

_DEFAULT_CHART_TYPE = "bar"


def _hint_chart_type(query: str) -> str:
    q = query.lower()
    for keywords, chart_type in _CHART_HINTS:
        if any(kw in q for kw in keywords):
            return chart_type
    return _DEFAULT_CHART_TYPE


# ────────────────────────────────────────────────────────────────────────────
# ROUTER HELPER  (used by rag_engine.router to detect graph intent)
# ────────────────────────────────────────────────────────────────────────────

_GRAPH_KEYWORDS = [
    "graph", "chart", "plot", "visuali", "bar chart", "line chart",
    "pie chart", "histogram", "scatter", "diagram of data",
    "show me a", "draw a", "create a chart", "generate a chart",
    "trend", "compare visually", "visual comparison",
]

def is_graph_request(query: str) -> bool:
    """
    Lightweight heuristic check — used as a FALLBACK inside the router.
    The LLM router should catch most cases; this guards the fallback path.
    """
    q = query.lower()
    return any(kw in q for kw in _GRAPH_KEYWORDS)


# ────────────────────────────────────────────────────────────────────────────
# LLM CALL HELPER
# ────────────────────────────────────────────────────────────────────────────

def _call_llm(
    prompt: str,
    system_prompt: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    task: str = "plan",
) -> str:
    payload = {
        "prompt":       prompt,
        "systemPrompt": system_prompt,
        "history":      [],
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "task":         task,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    for attempt in range(3):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                raw = resp.json().get("response") or ""
                return raw if isinstance(raw, str) else str(raw)
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"   ⚠️  GraphAgent LLM {resp.status_code}: {resp.text[:120]}")
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"   ⚠️  GraphAgent LLM failed: {e}")
    return ""


# ────────────────────────────────────────────────────────────────────────────
# JSON PARSE HELPER
# ────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[Any]:
    """Try JSON, then ast.literal_eval, then regex-extract first {...} block."""
    import ast

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # 1. Standard JSON
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Python repr (single-quoted dicts)
    try:
        return ast.literal_eval(clean)
    except (ValueError, SyntaxError):
        pass

    # 3. Extract first {...} block (handles leading/trailing noise)
    m = re.search(r'\{[\s\S]*\}', clean)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


# ────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ────────────────────────────────────────────────────────────────────────────

_PALETTE_BG = [
    "rgba(99,  102, 241, 0.75)",   # indigo
    "rgba(16,  185, 129, 0.75)",   # emerald
    "rgba(245, 158,  11, 0.75)",   # amber
    "rgba(239,  68,  68, 0.75)",   # red
    "rgba(59,  130, 246, 0.75)",   # blue
    "rgba(236,  72, 153, 0.75)",   # pink
    "rgba(139,  92, 246, 0.75)",   # violet
    "rgba(20,  184, 166, 0.75)",   # teal
    "rgba(249, 115,  22, 0.75)",   # orange
    "rgba(168, 85,  247, 0.75)",   # purple
]

_PALETTE_BORDER = [c.replace("0.75", "1") for c in _PALETTE_BG]

_LINE_COLORS = [
    "#6366f1", "#10b981", "#f59e0b", "#ef4444",
    "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6",
]


def _colors_for(n: int, chart_type: str) -> Tuple[List[str], List[str]]:
    if chart_type in ("pie", "doughnut"):
        bg     = (_PALETTE_BG     * ((n // len(_PALETTE_BG))     + 1))[:n]
        border = (_PALETTE_BORDER * ((n // len(_PALETTE_BORDER)) + 1))[:n]
    elif chart_type == "line":
        bg     = [_LINE_COLORS[i % len(_LINE_COLORS)] for i in range(n)]
        border = bg
    else:
        bg     = (_PALETTE_BG     * ((n // len(_PALETTE_BG))     + 1))[:n]
        border = (_PALETTE_BORDER * ((n // len(_PALETTE_BORDER)) + 1))[:n]
    return bg, border


# ────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ────────────────────────────────────────────────────────────────────────────

class GraphAgent:
    """
    Two-stage pipeline:
      Stage 1 — Extract: LLM reads document chunks and returns structured
                         data (labels + one or more series) as JSON.
      Stage 2 — Build:   Python assembles a Chart.js config dict, applying
                         the right type, colours, options, and axis labels.

    The returned dict is stored in AgentState["analytics"] with an extra key
      "type": "chart_config"
    so the frontend can branch on it and render a <canvas> instead of markdown.

    If extraction fails (no numeric data in documents, LLM error, etc.) the
    agent returns a graceful fallback with "type": "chart_error" and an
    explanation message the frontend can display instead.
    """

    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key  = api_key  or os.getenv("CF_API_KEY", "")
        self.base_url = base_url or os.getenv("CF_API_URL",
                        "https://bimloapi.medhelaliamin125.workers.dev")
        self.enabled  = bool(self.api_key)
        print(f"📈 GraphAgent ready [llm={'✅' if self.enabled else '❌'}]")

    # ── PUBLIC ENTRY POINT ────────────────────────────────────────────────

    def build_chart(
        self,
        query:   str,
        chunks:  List[Dict],
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Main method called by graph_node.

        Returns a dict with at minimum:
          {
            "type":         "chart_config",   # or "chart_error"
            "chart_type":   "bar"|"line"|"pie"|"scatter",
            "chart_js":     { ...Chart.js config... },
            "title":        "Human-readable chart title",
            "description":  "1–2 sentence explanation of what the chart shows",
            "sources":      ["filename1.pdf", ...],
          }
        """
        if not chunks:
            return self._error("No document content found to build a chart from.")

        if not self.enabled:
            return self._error(
                "Chart generation unavailable: CF_API_KEY is not set."
            )

        # Determine chart type from query + any explicit user axes hints
        chart_type = _hint_chart_type(query)
        axes_hint  = self._extract_axes_hint(query)

        print(f"   📊 GraphAgent: type={chart_type}, axes={axes_hint}")

        # Stage 1 — extract structured data from documents
        raw_data = self._extract_data(query, chunks, chart_type, axes_hint, language)
        if not raw_data:
            return self._error(
                "Could not find numeric or structured data in the documents "
                "that matches your chart request. Try asking about a specific "
                "metric, table, or set of values in the documents."
            )

        # Stage 2 — assemble Chart.js config
        chart_js = self._build_chartjs(raw_data, chart_type, query, language)
        if not chart_js:
            return self._error("Failed to assemble chart configuration.")

        # Collect source filenames
        sources = list(dict.fromkeys(
            c["metadata"].get("filename", "unknown") for c in chunks
            if c["metadata"].get("filename")
        ))

        return {
            "type":        "chart_config",
            "chart_type":  chart_type,
            "chart_js":    chart_js,
            "title":       raw_data.get("title", self._default_title(query)),
            "description": raw_data.get("description", ""),
            "sources":     sources,
            "raw_data":    raw_data,   # kept for debugging; frontend may ignore
        }

    # ── STAGE 1: DATA EXTRACTION ──────────────────────────────────────────

    def _extract_data(
        self,
        query:      str,
        chunks:     List[Dict],
        chart_type: str,
        axes_hint:  Dict[str, str],
        language:   str,
    ) -> Optional[Dict]:
        """
        Ask the LLM to read the document chunks and return:
          {
            "title":       "Chart title",
            "description": "What this chart shows",
            "x_label":     "X-axis label",
            "y_label":     "Y-axis label",
            "labels":      ["label1", "label2", ...],   // x-axis categories
            "datasets": [
              {
                "label": "Series name",
                "data":  [12.5, 34, 7, ...]
              },
              ...  // multiple datasets for grouped bars / multi-line charts
            ]
          }

        For scatter charts data points are {x, y} objects instead of scalars.
        """
        context = self._build_context(chunks)

        x_hint = axes_hint.get("x", "")
        y_hint = axes_hint.get("y", "")
        axes_instruction = ""
        if x_hint or y_hint:
            axes_instruction = (
                f"\nThe user specified axes: "
                f"{'X-axis = ' + x_hint if x_hint else ''}"
                f"{', ' if x_hint and y_hint else ''}"
                f"{'Y-axis = ' + y_hint if y_hint else ''}."
                f" Use these as your axis labels if the data supports it."
            )

        scatter_note = ""
        if chart_type == "scatter":
            scatter_note = (
                '\nFor scatter charts, "data" must be a list of {"x": number, "y": number} objects.'
            )

        system = (
            "You are a data extraction specialist. Your job is to read document "
            "text and extract structured numerical data for chart generation. "
            "You return ONLY valid JSON — no markdown, no explanation, no backticks."
        )

        prompt = f"""The user wants a {chart_type} chart. Their request: "{query}"{axes_instruction}

Read the document content below and extract all relevant numerical or categorical data.

REQUIRED JSON format — return EXACTLY this structure:
{{
  "title": "concise chart title in {language}",
  "description": "1-2 sentences explaining what the chart shows, in {language}",
  "x_label": "label for the x-axis (categories or time)",
  "y_label": "label for the y-axis (the measured values)",
  "labels": ["cat1", "cat2", "cat3", ...],
  "datasets": [
    {{
      "label": "series name",
      "data": [value1, value2, value3, ...]
    }}
  ]
}}
{scatter_note}

RULES:
- All values in "data" must be numbers (int or float), NOT strings.
- "labels" length must equal "data" length in each dataset.
- If you find multiple related series (e.g. budgeted vs actual, or multiple products),
  include them as separate objects in "datasets".
- If you only find one series, "datasets" should have exactly one object.
- Extract ALL available data points — do not truncate to fewer than what exists.
- If the document has a table, extract every row.
- If you cannot find any numerical data relevant to the request, return:
  {{"error": "no_data", "reason": "brief explanation"}}
- Return ONLY the JSON object. Nothing else.

DOCUMENTS:
{context}"""

        raw = _call_llm(prompt, system, self.api_key, self.base_url,
                        max_tokens=1500, temperature=0.0)
        if not raw:
            return None

        parsed = _parse_json(raw)
        if not parsed or not isinstance(parsed, dict):
            print(f"   ⚠️  GraphAgent: could not parse extraction response")
            print(f"      Raw: {raw[:200]}")
            return None

        if parsed.get("error") == "no_data":
            print(f"   ℹ️  GraphAgent: no data — {parsed.get('reason', '')}")
            return None

        # Validate required fields
        if not parsed.get("labels") or not parsed.get("datasets"):
            print(f"   ⚠️  GraphAgent: extraction missing labels/datasets")
            return None

        # Coerce all data values to numbers (guard against LLM returning strings)
        for ds in parsed.get("datasets", []):
            coerced = []
            for v in ds.get("data", []):
                if isinstance(v, dict):          # scatter {x, y}
                    coerced.append({
                        "x": float(v.get("x", 0)),
                        "y": float(v.get("y", 0)),
                    })
                else:
                    try:
                        coerced.append(float(str(v).replace(",", "").replace("%", "")))
                    except (ValueError, TypeError):
                        coerced.append(0.0)
            ds["data"] = coerced

        print(f"   ✅ GraphAgent extracted: {len(parsed['labels'])} labels, "
              f"{len(parsed['datasets'])} dataset(s)")
        return parsed

    # ── STAGE 2: CHART.JS CONFIG ASSEMBLY ────────────────────────────────

    def _build_chartjs(
        self,
        data:       Dict,
        chart_type: str,
        query:      str,
        language:   str,
    ) -> Optional[Dict]:
        """
        Assemble a complete Chart.js config dict from the extracted data.
        The config is JSON-serialisable and can be passed directly to
        new Chart(ctx, config) in the frontend.
        """
        labels   = data.get("labels", [])
        datasets = data.get("datasets", [])
        x_label  = data.get("x_label", "")
        y_label  = data.get("y_label", "")
        title    = data.get("title", self._default_title(query))

        if not labels or not datasets:
            return None

        n_datasets = len(datasets)
        is_pie     = chart_type in ("pie", "doughnut")
        is_scatter = chart_type == "scatter"
        is_line    = chart_type == "line"

        # ── Build dataset objects ──────────────────────────────────────────
        built_datasets = []
        for i, ds in enumerate(datasets):
            bg, border = _colors_for(
                len(labels) if is_pie else n_datasets, chart_type
            )
            entry: Dict[str, Any] = {
                "label":           ds.get("label", f"Series {i + 1}"),
                "data":            ds["data"],
                "backgroundColor": bg if is_pie else bg[i % len(bg)],
                "borderColor":     border if is_pie else border[i % len(border)],
                "borderWidth":     2,
            }
            if is_line:
                entry["fill"]        = False
                entry["tension"]     = 0.4
                entry["pointRadius"] = 4
                entry["pointHoverRadius"] = 6
            if is_scatter:
                entry["pointRadius"] = 5
                entry["pointHoverRadius"] = 8
            built_datasets.append(entry)

        # ── Scales ────────────────────────────────────────────────────────
        scales: Dict[str, Any] = {}
        if not is_pie:
            scales = {
                "x": {
                    "title": {
                        "display": bool(x_label),
                        "text":    x_label,
                        "color":   "#94a3b8",
                        "font":    {"size": 12, "weight": "500"},
                    },
                    "ticks": {
                        "color":    "#94a3b8",
                        "maxRotation": 45,
                        "font":    {"size": 11},
                    },
                    "grid": {"color": "rgba(148, 163, 184, 0.15)"},
                },
                "y": {
                    "title": {
                        "display": bool(y_label),
                        "text":    y_label,
                        "color":   "#94a3b8",
                        "font":    {"size": 12, "weight": "500"},
                    },
                    "ticks":      {"color": "#94a3b8", "font": {"size": 11}},
                    "grid":       {"color": "rgba(148, 163, 184, 0.15)"},
                    "beginAtZero": True,
                },
            }

        # ── Plugins (legend, tooltip, title) ─────────────────────────────
        plugins: Dict[str, Any] = {
            "legend": {
                "display":  n_datasets > 1 or is_pie,
                "position": "bottom" if is_pie else "top",
                "labels": {
                    "color":   "#cbd5e1",
                    "padding": 16,
                    "font":    {"size": 12},
                    "usePointStyle": True,
                },
            },
            "title": {
                "display": True,
                "text":    title,
                "color":   "#f1f5f9",
                "font":    {"size": 15, "weight": "600"},
                "padding": {"bottom": 18},
            },
            "tooltip": {
                "backgroundColor": "rgba(15, 23, 42, 0.9)",
                "titleColor":      "#f1f5f9",
                "bodyColor":       "#cbd5e1",
                "borderColor":     "rgba(99, 102, 241, 0.4)",
                "borderWidth":     1,
                "padding":         10,
                "cornerRadius":    8,
            },
        }

        # Pie slices: show label + value in tooltip
        if is_pie:
            plugins["tooltip"]["callbacks"] = {
                "label": "function(ctx){return ctx.label+': '+ctx.formattedValue;}"
            }

        # ── Final config ──────────────────────────────────────────────────
        config: Dict[str, Any] = {
            "type": chart_type,
            "data": {
                "labels":   labels,
                "datasets": built_datasets,
            },
            "options": {
                "responsive":          True,
                "maintainAspectRatio": True,
                "animation":           {"duration": 600, "easing": "easeInOutQuart"},
                "plugins":             plugins,
                "scales":              scales,
            },
        }

        # Horizontal bar
        if chart_type == "bar" and self._wants_horizontal(query):
            config["options"]["indexAxis"] = "y"

        return config

    # ── HELPERS ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict], max_chars_per_chunk: int = 1500) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            m = c.get("metadata", {})
            text = c.get("text", "")[:max_chars_per_chunk]
            parts.append(
                f"[Source {i} | {m.get('filename', 'unknown')} | {m.get('doc_type', '')}]\n{text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_axes_hint(query: str) -> Dict[str, str]:
        """
        Parse user-specified axis labels from natural language.
        Examples:
          "plot revenue on y axis and months on x" → {"x": "months", "y": "revenue"}
          "x = department, y = headcount"           → {"x": "department", "y": "headcount"}
        """
        hint: Dict[str, str] = {}
        q = query.lower()

        # Pattern: "X on the x axis" / "x-axis: X" / "x = X"
        for axis in ("x", "y"):
            patterns = [
                rf"(\w[\w\s]*?)\s+on\s+(?:the\s+)?{axis}[\s-]axis",
                rf"{axis}[\s-]axis\s*[=:]\s*([\w\s]+?)(?:\s+and|\s+y|\s+x|,|$)",
                rf"{axis}\s*=\s*([\w\s]+?)(?:\s+and|\s+y|\s+x|,|$)",
            ]
            for pat in patterns:
                m = re.search(pat, q)
                if m:
                    hint[axis] = m.group(1).strip()
                    break

        return hint

    @staticmethod
    def _wants_horizontal(query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in ["horizontal", "ranked", "rank", "top ", "bottom "])

    @staticmethod
    def _default_title(query: str) -> str:
        q = query.strip().rstrip("?").strip()
        # Strip leading "show me a / create a / generate a / plot" etc.
        q = re.sub(
            r"^(?:show\s+me\s+a?|create\s+a?|generate\s+a?|plot|draw\s+a?|make\s+a?)\s+",
            "", q, flags=re.IGNORECASE
        ).strip()
        return q[:80].title() if q else "Chart"

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "type":    "chart_error",
            "message": message,
        }


# ────────────────────────────────────────────────────────────────────────────
# GRAPH NODE METHOD — paste into RAGEngine class in rag_engine.py
# ────────────────────────────────────────────────────────────────────────────

GRAPH_NODE_METHOD = '''
    def graph_node(self, state: AgentState) -> AgentState:
        """
        Generate a Chart.js config from document data.

        Retrieval already ran (same path as analytics_node), so we have
        state["retrieved_chunks"].  GraphAgent reads them and returns a
        Chart.js config stored in state["analytics"].

        The frontend checks response["analytics"]["type"] == "chart_config"
        and renders a <canvas> with Chart.js instead of markdown text.
        """
        query  = state["query"]
        chunks = state["retrieved_chunks"]

        self._emit("graph_node", "📈", "Building chart from documents…")
        print(f"📈 graph_node → query={query[:80]}")

        # Ask the judge for language/tone (mirrors other nodes)
        history = state.get("conversation_history", [])
        history_texts = [f"{m[\'role\'].upper()}: {m[\'content\'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=chunks,
                                        conversation_history=history_texts)

        result = self.graph_agent.build_chart(
            query=query,
            chunks=chunks,
            language=plan.target_language,
        )

        if result.get("type") == "chart_error":
            # Fall back to a regular RAG answer explaining the issue
            print(f"   ⚠️  graph_node: {result[\'message\']}")
            fallback_answer = (
                f"I wasn\'t able to generate a chart: {result[\'message\']} "
                f"If your documents contain tables or numerical data, try being "
                f"more specific — for example: \'plot monthly revenue as a bar chart\'."
            )
            return {
                **state,
                "answer":     fallback_answer,
                "raw_answer": fallback_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result,
            }

        # Build a short narrative answer to accompany the chart
        description = result.get("description", "")
        title       = result.get("title", "Chart")
        sources_str = ", ".join(result.get("sources", []))
        answer = (
            f"Here is the **{title}** chart generated from your documents"
            f"{(' (' + sources_str + ')') if sources_str else ''}. "
            f"{description}"
        ).strip()

        print(f"   ✅ graph_node: chart ready — {title}")

        return {
            **state,
            "response_plan": plan,
            "answer":        answer,
            "raw_answer":    answer,
            "sources":       [],          # chart speaks for itself
            "confidence":    _confidence(chunks),
            "analytics":     result,
        }
'''

# ────────────────────────────────────────────────────────────────────────────
# ROUTER PROMPT ADDITION — paste into the routing_prompt in rag_engine.router
# ────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT_ADDITION = """
- graph: the user wants a chart, graph, or visual plot of data extracted from
  the documents — bar chart, line chart, pie chart, scatter plot, histogram,
  trend over time, comparison chart, etc. Triggered by words like: graph, chart,
  plot, visualize, bar chart, pie chart, line graph, histogram, show me a chart,
  compare visually, trend chart, distribution chart.
"""

# ────────────────────────────────────────────────────────────────────────────
# CHECK-RETRIEVAL ROUTE ADDITION — add inside _check_retrieval_route lambda
# ────────────────────────────────────────────────────────────────────────────

CHECK_RETRIEVAL_ROUTE_ADDITION = """
            if s["route"] == "graph":
                return "graph"
"""

# ────────────────────────────────────────────────────────────────────────────
# STATUS MSG ADDITION — add to _DEFAULT_STATUS_MSGS in RAGEngine
# ────────────────────────────────────────────────────────────────────────────

STATUS_MSG_ADDITION = {
    "graph_node": ("📈", "Building chart from documents…"),
}

# ────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json

    print("=== GraphAgent standalone test ===\n")

    # Minimal fake chunks so you can test extraction + config building
    fake_chunks = [
        {
            "text": (
                "Monthly Revenue Report 2024\n"
                "January:  $142,000\n"
                "February: $158,000\n"
                "March:    $134,000\n"
                "April:    $172,000\n"
                "May:      $189,000\n"
                "June:     $201,000\n"
            ),
            "metadata": {"filename": "revenue_2024.pdf", "doc_type": "financial"},
        },
        {
            "text": (
                "Department Headcount\n"
                "Engineering: 45\n"
                "Sales: 32\n"
                "Marketing: 18\n"
                "Operations: 27\n"
                "HR: 12\n"
            ),
            "metadata": {"filename": "headcount.xlsx", "doc_type": "hr"},
        },
    ]

    agent = GraphAgent()
    if not agent.enabled:
        print("⚠️  CF_API_KEY not set — set it in .env to run the full test.")
        print("   Testing color/config helpers only...\n")

        # Test colour helpers
        bg, border = _colors_for(6, "bar")
        print(f"Bar colours (6): {bg}\n")
        bg, border = _colors_for(4, "pie")
        print(f"Pie colours (4): {bg}\n")

        # Test axes extraction
        queries = [
            "plot revenue on y axis and months on x axis",
            "bar chart with department on x and headcount on y",
            "show me a horizontal bar chart of top departments",
            "chart comparing Jan vs Feb vs March",
        ]
        for q in queries:
            ct   = _hint_chart_type(q)
            axes = GraphAgent._extract_axes_hint(q)
            horiz = GraphAgent._wants_horizontal(q)
            print(f"  Q: {q}")
            print(f"     type={ct}, axes={axes}, horizontal={horiz}\n")
    else:
        result = agent.build_chart(
            query="show me a line chart of monthly revenue over 2024",
            chunks=[fake_chunks[0]],
        )
        print(json.dumps(result, indent=2, default=str))
