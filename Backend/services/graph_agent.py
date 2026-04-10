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
    # English
    "graph", "chart", "plot", "visuali", "bar chart", "line chart",
    "pie chart", "histogram", "scatter", "diagram of data",
    "show me a", "draw a", "create a chart", "generate a chart",
    "trend", "compare visually", "visual comparison",
    # French
    "graphique", "graphe", "diagramme", "courbe", "histogramme",
    "camembert", "nuage de points", "tracer", "visualis", "montrer un",
    "créer un graphique", "faire un graphique", "générer un graphique",
    "afficher un", "comparaison visuelle",
    # Arabic
    "رسم بياني", "مخطط", "رسم", "تصور", "بياني", "عمودي", "دائري",
    "خطي", "انتشار", "مقارنة بيانية",
    # Spanish
    "gráfico", "gráfica", "diagrama", "histograma", "pastel", "dispersión",
    "visualizar", "trazar", "mostrar gráfico", "crear gráfico", "comparación visual",
    # German
    "diagramm", "grafik", "kurve", "balkendiagramm", "kreisdiagramm",
    "histogramm", "visualisier", "zeig mir ein", "erstell ein diagramm",
    # Italian
    "grafico", "diagramma", "istogramma", "torta", "dispersione",
    "visualizza", "traccia", "crea grafico",
    # Portuguese
    "gráfico", "diagrama", "histograma", "pizza", "dispersão",
    "visualizar", "traçar", "criar gráfico",
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
    api_key: str,       # kept for signature compatibility — routing handled inside call_llm
    base_url: str,      # kept for signature compatibility
    max_tokens: int = 1200,
    temperature: float = 0.0,
    task: str = "plan",
) -> str:
    from llm_client import call_llm
    return call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        history=[],
        max_tokens=max_tokens,
        temperature=temperature,
        task=task,
    )


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
        query:              str,
        chunks:             List[Dict],
        language:           str = "en",
        skip_clarification: bool = False,
    ) -> Dict[str, Any]:
        """
        Main method called by graph_node.

        Returns a dict with type one of:
          "chart_config"       — ready Chart.js config + interpretation
          "chart_clarification"— vague query; returns metric groups for user to pick
          "chart_error"        — could not generate

        skip_clarification=True bypasses Stage 0 entirely — used when the user
        has already responded to a clarification prompt (to prevent looping).
        """
        if not chunks:
            return self._error("No document content found to build a chart from.")

        if not self.enabled:
            return self._error(
                "Chart generation unavailable: CF_API_KEY is not set."
            )

        # ── Stage 0: check if query is vague and needs clarification ─────────
        # Skipped when the user already answered a clarification (skip_clarification=True)
        if not skip_clarification and self._is_vague_request(query):
            groups = self._discover_metric_groups(query, chunks, language)
            if groups and len(groups) > 1:
                # ── Validate each group: only show options that actually produce a chart ──
                # This prevents the user clicking an option and getting "no data found".
                validated_groups = []
                for grp in groups:
                    hint = grp.get("hint", "")
                    if not hint:
                        continue
                    try:
                        probe = self._extract_data(
                            query=hint,
                            chunks=chunks,
                            chart_type=_hint_chart_type(hint),
                            axes_hint=self._extract_axes_hint(hint),
                            language=language,
                        )
                        if probe is not None:
                            validated_groups.append(grp)
                            print(f"      ✅ group validated: {grp['label']!r}")
                        else:
                            print(f"      ❌ group dropped (no extractable data): {grp['label']!r}")
                    except Exception as _ve:
                        print(f"      ⚠️  group probe error for {grp['label']!r}: {_ve}")

                if len(validated_groups) > 1:
                    print(f"   🤔 GraphAgent: vague query — {len(validated_groups)}/{len(groups)} groups validated")
                    group_names = ", ".join(g["label"] for g in validated_groups[:3])
                    clarif_question = (
                        f"I found several distinct data sets in your documents that I could chart: "
                        f"{group_names}{'...' if len(validated_groups) > 3 else ''}. "
                        f"Which one would you like to visualize?"
                    )
                    return {
                        "type":     "chart_clarification",
                        "question": clarif_question,
                        "groups":   validated_groups,
                    }
                elif len(validated_groups) == 1:
                    # Only one valid group — build it silently without asking
                    print(f"   📊 GraphAgent: only 1 valid group after validation — building directly")
                    query = validated_groups[0]["hint"]
                # else: nothing validated — fall through to normal extraction with original query

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

        # Stage 3 — generate interpretation paragraph
        interpretation = self._interpret_chart(raw_data, query, language)

        # Collect source filenames
        sources = list(dict.fromkeys(
            c["metadata"].get("filename", "unknown") for c in chunks
            if c["metadata"].get("filename")
        ))

        return {
            "type":           "chart_config",
            "chart_type":     chart_type,
            "chart_js":       chart_js,
            "title":          raw_data.get("title", self._default_title(query)),
            "description":    raw_data.get("description", ""),
            "interpretation": interpretation,
            "sources":        sources,
            "raw_data":       raw_data,
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
            "You are a data extraction specialist for Bimlo Copilot — the AI assistant of BIMLO TECHNOLOGIE, "
            "a company specialising in BIM engineering, telecom infrastructure, and DeepTwin AI. "
            "Your job is to read document text and extract structured numerical data for chart generation. "
            "You return ONLY valid JSON — no markdown, no explanation, no backticks."
        )

        prompt = f"""The user wants a {chart_type} chart. Their request: "{query}"{axes_instruction}

Read the document content below and extract all relevant numerical or categorical data.

IMPORTANT — the document may contain data in ANY of these formats:
- Markdown/text tables: columns separated by | or spaces, with header rows
- PDF-extracted text: columns may appear as whitespace-separated values on each line
- Key-value pairs: "Metric: value" or "Metric = value" format
- Paragraph mentions: numbers mentioned inline like "Site A had 45.2 Mbps throughput"
- Comma/semicolon-separated lists of values
- Indented or aligned columns (align by character position)

Extract ALL numeric data relevant to the request, regardless of format.

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
- All values in "data" must be numbers (int or float), NOT strings. Strip units (Mbps, dB, %, ms) and convert.
- "labels" length must equal "data" length in each dataset.
- If the document has multiple related columns (e.g. Downlink, Uplink, Latency), include each as a SEPARATE dataset.
- If you find a table, extract EVERY row — do not truncate.
- For PDF text that looks like aligned columns (spaces between values), parse each line as a row.
- If the document mentions values inline in sentences, still extract them.
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


    # ── STAGE 0: VAGUENESS + METRIC DISCOVERY ────────────────────────────

    _VAGUE_PATTERNS = [
        # English
        r"^(show|make|create|generate|draw|plot|give me|build)\s+(me\s+)?(a\s+)?(chart|graph|plot|visual)$",
        r"^(chart|graph|plot|visuali[zs]e)\s+(the\s+)?(data|documents?|everything|metrics|numbers|stats|statistics)$",
        r"^(what|show).*(look like|visuali[zs]|chart|graph)",
        r"^(all|every).*(metric|number|stat|data|value)",
        # French
        r"^(montre|cr[eé]e|g[eé]n[eè]re|fais|trace|dessine)\s+(moi\s+)?(un\s+)?(graphique|diagramme|courbe|visuel)$",
        r"^(graphique|diagramme|visualis[ae])\s+(les?\s+)?(donn[eé]es?|documents?|m[eé]triques?|statistiques?)$",
        r"^(tout|tous|toutes)\s+(les?\s+)?(m[eé]trique|donn[eé]e|stat|nombre|valeur)",
        # Arabic
        r"^(أنشئ|اصنع|ارسم|أظهر|أنشأ)\s+(لي\s+)?(رسم|مخطط|بياني)$",
        r"^(كل|جميع)\s+(البيانات|المقاييس|الأرقام|الإحصاءات)",
        # Spanish
        r"^(muestra|crea|genera|haz|traza|dibuja)\s+(me\s+)?(una?\s+)?(gráfica?|diagrama|visual)$",
        r"^(gráfica?|diagrama|visualiza)\s+(los?\s+)?(datos?|documentos?|métricas?|estadísticas?)$",
        # German
        r"^(zeig|erstell|generier|mal|zeichne)\s+(mir\s+)?(ein\s+)?(diagramm|grafik|chart|visual)$",
        r"^(alle?|jede[rs]?)\s+(daten|metriken|zahlen|statistiken)",
    ]

    def _is_vague_request(self, query: str) -> bool:
        """
        Returns True when the query doesn't specify WHAT to chart.
        Heuristic: short query with no domain nouns, or matches known vague patterns.
        """
        import re as _re
        q = query.strip().lower().rstrip("?.")

        # Explicit vague patterns
        for pat in self._VAGUE_PATTERNS:
            if _re.search(pat, q):
                return True

        # Short query (≤6 words) with no numeric hint and no specific noun
        words = q.split()
        if len(words) <= 6:
            # If it mentions a specific metric word it's not vague (multilingual)
            specific_hints = [
                # English — generic
                "revenue", "cost", "budget", "salary", "temperature", "speed",
                "latency", "throughput", "loss", "power", "load", "rate",
                "count", "total", "average", "percent", "ratio", "score",
                "traffic", "bandwidth", "frequency", "voltage", "current",
                "pressure", "distance", "weight", "height", "duration",
                # Telecom / network (added)
                "downlink", "uplink", "rsrp", "rsrq", "sinr", "rssi",
                "insertion", "fiber", "fibre", "strand", "strands", "trench",
                "trenching", "span", "attenuation", "signal", "sector",
                "bbu", "rrh", "antenna", "antennae", "mast", "tower",
                "node", "site", "sites", "base station", "coverage",
                "packet", "jitter", "availability", "alarm", "alarms",
                "otdr", "odf", "splitter", "amplifier", "transceiver",
                # Power / electrical (added)
                "dc", "ac", "watt", "kilowatt", "ampere", "volt", "battery",
                "autonomy", "capacity", "utilisation", "utilization",
                "shelter", "hvac", "cooling",
                # BIM / construction / project management
                "clash", "clashes", "conflict", "conflicts", "milestone", "milestones",
                "completion", "schedule", "timeline", "deadline", "progress", "status",
                "coordination", "model", "structural", "architectural", "mep",
                "framing", "simulation", "review", "inspection", "phase", "phases",
                "submittal", "rfi", "punch", "defect", "issue", "issues",
                "quantity", "quantities", "takeoff", "area", "volume",
                "storey", "storeys", "floor", "floors", "material", "materials",
                "element", "elements", "wall", "slab", "column", "beam",
                # French
                "revenu", "coût", "budget", "salaire", "température", "vitesse",
                "latence", "débit", "perte", "puissance", "charge", "taux",
                "total", "moyenne", "pourcentage", "fréquence", "tension",
                "pression", "distance", "poids", "durée", "trafic",
                "fibre", "brin", "portée", "tranchée", "batterie",
                # Arabic
                "إيراد", "تكلفة", "ميزانية", "راتب", "سرعة", "كمون",
                "إنتاجية", "طاقة", "حمل", "معدل", "مجموع", "متوسط",
                "نسبة", "تردد", "ضغط", "مسافة", "وزن", "مدة",
                # Spanish
                "ingreso", "costo", "presupuesto", "salario", "velocidad",
                "latencia", "rendimiento", "pérdida", "potencia", "carga",
                "tasa", "total", "promedio", "porcentaje", "frecuencia",
                "tensión", "presión", "distancia", "peso", "duración",
                # German
                "umsatz", "kosten", "budget", "gehalt", "geschwindigkeit",
                "latenz", "durchsatz", "verlust", "leistung", "last",
                "rate", "gesamt", "durchschnitt", "prozent", "frequenz",
                # Italian / Portuguese (shared roots)
                "entrate", "costo", "stipendio", "velocità", "latenza",
                "rendimento", "potenza", "carico", "tasso", "totale",
                "receita", "custo", "salário", "taxa", "frequência",
            ]
            if not any(h in q for h in specific_hints):
                return True

        return False

    def _discover_metric_groups(
        self,
        query: str,
        chunks: List[Dict],
        language: str,
    ) -> List[Dict[str, str]]:
        """
        Ask the LLM to scan the document chunks and return distinct metric groups
        the user could chart — each group is cohesive (same unit / same domain).

        Returns a list like:
          [
            {"label": "Network Performance", "description": "Throughput, latency, packet loss", "hint": "chart network performance metrics"},
            {"label": "Power & Infrastructure", "description": "Power load, cable length, equipment specs", "hint": "chart power and infrastructure metrics"},
          ]
        """
        context = self._build_context(chunks, max_chars_per_chunk=1200)

        system = (
            "You are a data analyst for Bimlo Copilot — the AI assistant of BIMLO TECHNOLOGIE (BIM engineering, telecom infrastructure, DeepTwin AI). "
            "You help users choose what to chart from their BIM, construction, or telecom documents. "
            "You return ONLY valid JSON — no markdown, no explanation, no backticks."
        )

        prompt = f"""The user asked: "{query}"

Scan the document content below and identify 2-5 DISTINCT groups of numeric data that could each make a meaningful chart.
Each group must share the same unit or domain (e.g. all throughput values, all power readings, all equipment counts).

Return ONLY a JSON array — no markdown, no backticks, no explanation:
[
  {{
    "label": "Short group name (2-4 words, in {language})",
    "description": "Comma-separated list of the EXACT metric names as they appear in the document",
    "hint": "A very specific chart request using the exact metric and column names from the document, e.g. 'chart downlink throughput in Mbps per site as a bar chart' or 'plot insertion loss per span as a line chart'",
    "source_file": "the filename this data comes from, exactly as shown in the document header"
  }}
]

CRITICAL rules for "hint":
- Use the EXACT metric name as it appears in the document (copy the column header or field name verbatim)
- Include the unit if present (Mbps, dB, W, m, etc.)
- Include the grouping dimension (per site, per span, by section, etc.)
- Include a chart type (bar chart, line chart, pie chart)
- The hint must be specific enough that a chart can be built from it WITHOUT any further clarification
- BAD example: "chart network performance metrics" (too vague — will fail)
- GOOD example: "chart downlink throughput in Mbps per site as a bar chart from site_survey.pdf" (exact, buildable, names the file)

Rules:
- 2 to 5 groups maximum
- Each group must be internally coherent (same unit, same domain, or same time series)
- "label" must be short and clear — the user will click it as a button — include the filename if it helps identify which doc
- "source_file" must be the EXACT filename from the document header (e.g. "network_report.pdf")
- Return ONLY the JSON array. Nothing else.

DOCUMENTS:
{context}"""

        raw = _call_llm(prompt, system, self.api_key, self.base_url,
                        max_tokens=600, temperature=0.2)
        if not raw:
            return []

        parsed = _parse_json(raw)
        if not isinstance(parsed, list):
            # Try to unwrap if nested
            if isinstance(parsed, dict) and "groups" in parsed:
                parsed = parsed["groups"]
            else:
                print(f"   ⚠️  GraphAgent._discover_metric_groups: unexpected shape")
                return []

        # Validate each entry
        valid = []
        for item in parsed:
            if isinstance(item, dict) and item.get("label") and item.get("hint"):
                source_file = str(item.get("source_file", "")).strip()
                description = str(item.get("description", ""))[:120]
                # Append filename to description if available so the frontend shows it
                if source_file and source_file not in description:
                    description = f"{description} — {source_file}".strip(" —")
                valid.append({
                    "label":       str(item["label"])[:60],
                    "description": description,
                    "hint":        str(item["hint"])[:200],
                })
        return valid[:5]

    # ── STAGE 3: INTERPRETATION ───────────────────────────────────────────

    def _interpret_chart(
        self,
        raw_data: Dict,
        query:    str,
        language: str,
    ) -> str:
        """
        Generate a 2-4 sentence human-readable interpretation of the chart:
        key takeaways, outliers, trends — not just a description of what it shows.
        """
        labels   = raw_data.get("labels", [])
        datasets = raw_data.get("datasets", [])
        title    = raw_data.get("title", "")

        if not labels or not datasets:
            return ""

        # Build a compact data summary for the LLM
        data_lines = []
        for ds in datasets[:4]:
            series_name = ds.get("label", "Series")
            values = ds.get("data", [])
            pairs  = ", ".join(
                f"{labels[i]}: {values[i]}"
                for i in range(min(len(labels), len(values)))
            )
            data_lines.append(f"  {series_name}: {pairs}")
        data_summary = "\n".join(data_lines)

        system = (
            "You are a data analyst for Bimlo Copilot — the AI assistant of BIMLO TECHNOLOGIE "
            "(BIM engineering, telecom infrastructure, DeepTwin AI). "
            "You give sharp, specific insights relevant to BTP/construction and telecom professionals — not generic observations. "
            "Respond in plain prose with no markdown, no bullet points."
        )

        prompt = f"""Chart title: "{title}"
User request: "{query}"

Data:
{data_summary}

Write a 2-4 sentence interpretation in {language}. Focus on:
- The most notable value (highest/lowest/outlier)
- Any clear trend or pattern
- One practical implication if relevant

Rules:
- Be specific — reference actual values and labels from the data
- No bullet points, no markdown, no headers
- Do NOT start with "This chart shows" or "The chart displays"
- Output ONLY the interpretation text"""

        raw = _call_llm(prompt, system, self.api_key, self.base_url,
                        max_tokens=200, temperature=0.4)
        return raw.strip() if raw else ""

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
        Generate a Chart.js config from document data, alongside a full RAG
        narrative answer.  The chart is a bonus visual — the text answer is primary.

        Flow:
          1. Run GraphAgent.build_chart() to get chart config (or clarification)
          2. If clarification needed → return prose question + option groups, no chart yet
          3. If chart ready → synthesise a full narrative answer via LLM, attach chart
          4. Return both: answer (narrative prose) + analytics (chart config)
        """
        query  = state["query"]
        chunks = state["retrieved_chunks"]

        self._emit("graph_node", "📈", "Building chart from documents…")
        print(f"📈 graph_node → query={query[:80]}")

        history = state.get("conversation_history", [])
        history_texts = [f"{m[\'role\'].upper()}: {m[\'content\'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=chunks,
                                        conversation_history=history_texts)

        # Detect if this query is a clarification follow-up by checking whether
        # the most recent assistant turn was a chart_clarification.  In that case
        # skip Stage 0 so the agent doesn't re-ask the same question.
        prev_messages = state.get("conversation_history", [])
        # The analytics dict is attached to the state when coming from a prior
        # clarification; alternatively check if force_route was set by the frontend.
        was_clarification_response = (
            state.get("force_route") == "graph"
            and any(
                m.get("role") == "assistant"
                for m in prev_messages[-2:]
            )
        )

        result = self.graph_agent.build_chart(
            query=query,
            chunks=chunks,
            language=plan.target_language,
            skip_clarification=was_clarification_response,
        )

        # ── Clarification needed — vague query or multiple data sets found ────
        if result.get("type") == "chart_clarification":
            clarif_answer = result.get("question", "Which data would you like to chart?")
            return {
                **state,
                "response_plan": plan,
                "answer":        clarif_answer,
                "raw_answer":    clarif_answer,
                "sources":       [],
                "confidence":    0.5,
                "analytics":     result,
            }

        # ── Chart error — fall back to a regular narrative answer ─────────────
        if result.get("type") == "chart_error":
            print(f"   ⚠️  graph_node: {result['message']}")
            _example_hint = _build_chart_example_hint(chunks)
            fallback_answer = (
                f"I wasn't able to generate a chart: {result['message']} "
                f"Make sure your documents contain tables or columns of numeric data, "
                f"then try something specific — for example: '{_example_hint}'."
            )
            return {
                **state,
                "answer":     fallback_answer,
                "raw_answer": fallback_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result,
            }

        # ── Chart ready — synthesise a real narrative answer alongside it ─────
        title       = result.get("title", "Chart")
        description = result.get("description", "")
        sources_str = ", ".join(result.get("sources", []))
        interp      = result.get("interpretation", "")

        # Build document context for the narrative LLM call
        context_parts = []
        for i, c in enumerate(chunks, 1):
            m = c.get("metadata", {})
            context_parts.append(
                f"[Source {i} | {m.get(\'filename\', \'unknown\')}]\n{c.get(\'text\', \'\')[:800]}"
            )
        context_text = "\n\n".join(context_parts)

        narrative_prompt = (
            f"The user asked: \\"{query}\\"\n\n"
            f"You have just generated a **{title}** chart"
            f"{(\' from \' + sources_str) if sources_str else \'\'}.\n"
            f"Chart description: {description}\n"
            f"Key insight from the data: {interp}\n\n"
            f"Write a concise, natural answer (3-6 sentences) that:\n"
            f"1. Directly answers what the user asked in prose\n"
            f"2. Mentions the most important finding or number from the data\n"
            f"3. Notes that a chart has been generated below for a visual breakdown\n\n"
            f"Rules:\n"
            f"- Do NOT just describe what the chart shows — give a real answer\n"
            f"- Do NOT use bullet points\n"
            f"- Write in {plan.target_language}\n\n"
            f"Document context:\n{context_text}"
        )

        sys_prompt = (
            f"You are a helpful document assistant. Answer the user\'s question in "
            f"{plan.target_language} with a {plan.target_tone} tone. Be concise and direct."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": narrative_prompt},
        ]

        if self.llm.enabled:
            narrative = self.llm.chat(messages, max_tokens=300, temperature=0.4)
        else:
            narrative = (
                f"Here is the **{title}** chart"
                f"{(\' (\' + sources_str + \')\') if sources_str else \'\'}. "
                f"{description}"
            ).strip()

        print(f"   ✅ graph_node: chart + narrative ready — {title}")

        return {
            **state,
            "response_plan": plan,
            "answer":        narrative,
            "raw_answer":    narrative,
            "sources":       [],
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
  trend over time, comparison chart, etc.
  This route must trigger regardless of the language the user writes in.
  Examples across languages:
  EN: graph, chart, plot, visualize, bar chart, pie chart, histogram, show me a chart
  FR: graphique, diagramme, courbe, histogramme, camembert, visualiser, tracer, montrer un graphique
  AR: رسم بياني, مخطط, رسم, تصور بياني
  ES: gráfico, diagrama, histograma, visualizar, trazar
  DE: Diagramm, Grafik, Balkendiagramm, Kreisdiagramm, visualisieren
  IT: grafico, diagramma, visualizzare, tracciare
  PT: gráfico, diagrama, visualizar, traçar
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