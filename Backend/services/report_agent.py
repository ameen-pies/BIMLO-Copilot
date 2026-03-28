"""
report_agent.py — Smart Report Generation Agent

Architecture
────────────
• Pure FastAPI router  — no hardcoding, no standalone generate-report endpoint duplication.
• Uses RAG context     — pulls retrieved_chunks and conversation_history from SharedContext
                          (set by main.py after every /query call) so it reasons over the same
                          grounded content as every other agent.
• LLM-driven           — structure, language, tone, depth are all decided by the LLM; the agent
                          only sets up framing.
• Communicates context — SharedContext is a simple in-process store keyed by session_id.
                          Any agent that runs after /query (suggest, report, autocomplete…) reads
                          the same chunks + history without a second retrieval pass.
• Downloads            — ReportLab PDF (always works, no subprocess) and Markdown (.md).
                          PDF rendering path: ReportLab primary → pdflatex fallback.
                          ReportLab is always available so the PDF is never empty.

Routes
──────
POST   /reports                          — generate a new report
GET    /reports                          — list all report summaries
GET    /reports/{report_id}              — get one full report
PATCH  /reports/{report_id}             — edit / regenerate a section
DELETE /reports/{report_id}             — delete
GET    /reports/{report_id}/download    — download as ?fmt=pdf|md
POST   /reports/{report_id}/restore     — restore a previous version
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import subprocess
import tempfile
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

# ── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, ListFlowable, ListItem, Image,
)
from reportlab.platypus import Frame, PageTemplate
from reportlab.lib.utils import ImageReader

# ── LLM client (shared with all other agents) ─────────────────────────────────
try:
    from llm_client import call_llm
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    def call_llm(prompt, system_prompt="", history=None, max_tokens=2000,
                 temperature=0.3, task="synthesise"):
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# BRAND / STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

_BRAND_BLUE    = colors.HexColor("#1a56db")
_BRAND_DARK    = colors.HexColor("#1a1a2e")
_BRAND_LIGHT   = colors.HexColor("#eff6ff")
_GRAY_TEXT     = colors.HexColor("#6b7280")
_GRAY_BODY     = colors.HexColor("#1a1a1a")
_GRAY_BORDER   = colors.HexColor("#e5e7eb")
_GRAY_ALT_ROW  = colors.HexColor("#f9fafb")
_HEADING3_COL  = colors.HexColor("#374151")


def _build_styles() -> Dict[str, ParagraphStyle]:
    """Return a dict of named ParagraphStyles for the report PDF."""
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=26, fontName="Helvetica-Bold",
            textColor=_BRAND_DARK, alignment=TA_CENTER, leading=32, spaceAfter=10,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontSize=10, fontName="Helvetica",
            textColor=_GRAY_TEXT, alignment=TA_CENTER, spaceAfter=4,
        ),
        "toc_heading": ParagraphStyle(
            "toc_heading", fontSize=14, fontName="Helvetica-Bold",
            textColor=_BRAND_DARK, spaceAfter=10, spaceBefore=4,
        ),
        "toc_entry": ParagraphStyle(
            "toc_entry", fontSize=10, fontName="Helvetica",
            textColor=_GRAY_BODY, leading=16, leftIndent=0,
        ),
        "h1": ParagraphStyle(
            "h1", fontSize=17, fontName="Helvetica-Bold",
            textColor=_BRAND_DARK, spaceAfter=4, spaceBefore=14, leading=22,
        ),
        "h2": ParagraphStyle(
            "h2", fontSize=13, fontName="Helvetica-Bold",
            textColor=_BRAND_BLUE, spaceAfter=4, spaceBefore=12, leading=17,
        ),
        "h3": ParagraphStyle(
            "h3", fontSize=11, fontName="Helvetica-Bold",
            textColor=_HEADING3_COL, spaceAfter=3, spaceBefore=8, leading=14,
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, fontName="Helvetica",
            textColor=_GRAY_BODY, leading=15, spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontSize=10, fontName="Helvetica",
            textColor=_GRAY_BODY, leading=14, spaceAfter=3, leftIndent=16,
        ),
        "table_header": ParagraphStyle(
            "table_header", fontSize=9, fontName="Helvetica-Bold",
            textColor=colors.white, leading=12,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", fontSize=9, fontName="Helvetica",
            textColor=_GRAY_BODY, leading=12,
        ),
        "blockquote": ParagraphStyle(
            "blockquote", fontSize=10, fontName="Helvetica-Oblique",
            textColor=_HEADING3_COL, leftIndent=20, rightIndent=10,
            leading=14, spaceAfter=6, spaceBefore=4,
        ),
        "source": ParagraphStyle(
            "source", fontSize=8, fontName="Helvetica",
            textColor=_GRAY_TEXT, leading=12, spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "footer", fontSize=8, fontName="Helvetica",
            textColor=_GRAY_TEXT, alignment=TA_CENTER,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN → REPORTLAB STORY
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_inline(text: str) -> str:
    """
    Convert markdown inline markup → ReportLab XML markup.
    Order matters: bold+italic before bold/italic to avoid partial matches.
    Also escapes raw & < > that aren't part of our injected tags.
    """
    # Escape bare XML chars first (before we inject our own tags)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold + italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    return text


def _parse_table(table_lines: List[str], styles: Dict) -> Optional[Table]:
    """Parse markdown table lines into a ReportLab Table."""
    rows = []
    for line in table_lines:
        stripped = line.strip()
        # Skip separator rows like |---|---|
        if re.match(r'^\|[\s\-\|:]+\|$', stripped):
            continue
        cells = [c.strip() for c in stripped.strip('|').split('|')]
        rows.append(cells)

    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [''] * (col_count - len(r)) for r in rows]

    # Build ReportLab table data
    table_data = []
    for ri, row in enumerate(rows):
        if ri == 0:
            table_data.append([
                Paragraph(f'<b>{_parse_inline(c)}</b>', styles["table_header"])
                for c in row
            ])
        else:
            table_data.append([
                Paragraph(_parse_inline(c), styles["table_cell"])
                for c in row
            ])

    page_width = A4[0] - 5 * cm   # usable width
    col_width = page_width / col_count

    t = Table(table_data, colWidths=[col_width] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        # Header row
        ('BACKGROUND',   (0, 0), (-1, 0), _BRAND_BLUE),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        # All rows
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('GRID',         (0, 0), (-1, -1), 0.4, _GRAY_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, _GRAY_ALT_ROW]),
        ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
    ]))
    return t


def _md_to_story(content: str, styles: Dict) -> List:
    """
    Convert a Markdown string into a list of ReportLab Flowables.
    Handles: H1/H2/H3, paragraphs, bullet lists, numbered lists,
    markdown tables, blockquotes, fenced code blocks, horizontal rules.
    """
    story = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Empty line ────────────────────────────────────────────────────────
        if not stripped:
            story.append(Spacer(1, 3))
            i += 1
            continue

        # ── Fenced code block ─────────────────────────────────────────────────
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            # Use a light-grey box; Courier font
            code_para = Paragraph(
                '<font name="Courier" size="8">' +
                code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>") +
                '</font>',
                ParagraphStyle("code", fontName="Courier", fontSize=8, leading=12,
                               backColor=colors.HexColor("#f3f4f6"),
                               leftIndent=8, rightIndent=8,
                               borderPadding=(6, 6, 6, 6), spaceAfter=6),
            )
            story.append(code_para)
            continue

        # ── Horizontal rule ───────────────────────────────────────────────────
        if re.match(r'^[-*_]{3,}$', stripped):
            story.append(HRFlowable(width="100%", thickness=0.5, color=_GRAY_BORDER,
                                    spaceBefore=4, spaceAfter=4))
            i += 1
            continue

        # ── H1 ───────────────────────────────────────────────────────────────
        if stripped.startswith("# ") and not stripped.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(_parse_inline(stripped[2:]), styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=1, color=_GRAY_BORDER,
                                    spaceAfter=5))
            i += 1
            continue

        # ── H2 ───────────────────────────────────────────────────────────────
        if stripped.startswith("## ") and not stripped.startswith("### "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(_parse_inline(stripped[3:]), styles["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=_GRAY_BORDER, spaceAfter=4))
            i += 1
            continue

        # ── H3 ───────────────────────────────────────────────────────────────
        if stripped.startswith("### "):
            story.append(Spacer(1, 5))
            story.append(Paragraph(_parse_inline(stripped[4:]), styles["h3"]))
            i += 1
            continue

        # ── Markdown table ────────────────────────────────────────────────────
        if stripped.startswith("|") and "|" in stripped:
            table_lines = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            t = _parse_table(table_lines, styles)
            if t:
                story.append(Spacer(1, 6))
                story.append(t)
                story.append(Spacer(1, 6))
            continue

        # ── Bullet list ───────────────────────────────────────────────────────
        if re.match(r'^[\s]*[-*+]\s', line):
            items = []
            while i < len(lines) and re.match(r'^[\s]*[-*+]\s', lines[i]):
                text = re.sub(r'^[\s]*[-*+]\s', '', lines[i]).strip()
                items.append(
                    ListItem(Paragraph(_parse_inline(text), styles["bullet"]),
                             leftIndent=18, bulletColor=_BRAND_BLUE)
                )
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="•",
                                      leftIndent=10, spaceBefore=3, spaceAfter=3))
            continue

        # ── Numbered list ─────────────────────────────────────────────────────
        if re.match(r'^[\s]*\d+\.\s', line):
            items = []
            while i < len(lines) and re.match(r'^[\s]*\d+\.\s', lines[i]):
                text = re.sub(r'^[\s]*\d+\.\s', '', lines[i]).strip()
                items.append(
                    ListItem(Paragraph(_parse_inline(text), styles["bullet"]),
                             leftIndent=18)
                )
                i += 1
            story.append(ListFlowable(items, bulletType="1",
                                      leftIndent=10, spaceBefore=3, spaceAfter=3))
            continue

        # ── Blockquote ────────────────────────────────────────────────────────
        if stripped.startswith("> "):
            story.append(Paragraph(_parse_inline(stripped[2:]), styles["blockquote"]))
            i += 1
            continue

        # ── Normal paragraph ──────────────────────────────────────────────────
        story.append(Paragraph(_parse_inline(stripped), styles["body"]))
        i += 1

    return story


# ═══════════════════════════════════════════════════════════════════════════════
# PDF BUILDER  (ReportLab — no subprocess, always works)
# ═══════════════════════════════════════════════════════════════════════════════

def _on_page(canvas, doc, title: str, date_str: str):
    """Draw header + footer on every page except the cover."""
    page_num = doc.page
    if page_num == 1:
        return   # cover page — no header/footer

    canvas.saveState()
    page_w, page_h = A4

    # Header line
    canvas.setStrokeColor(_GRAY_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2.5 * cm, page_h - 2.0 * cm, page_w - 2.5 * cm, page_h - 2.0 * cm)

    # Header title
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_GRAY_TEXT)
    canvas.drawString(2.5 * cm, page_h - 1.7 * cm, title[:80])
    canvas.drawRightString(page_w - 2.5 * cm, page_h - 1.7 * cm, date_str)

    # Footer line
    canvas.line(2.5 * cm, 1.8 * cm, page_w - 2.5 * cm, 1.8 * cm)

    # Footer text
    canvas.drawCentredString(page_w / 2, 1.4 * cm, f"Page {page_num}  ·  Generated by Bimlo Copilot")

    canvas.restoreState()


def _chart_to_png_pdf(chart_cfg: Dict) -> Optional[bytes]:
    """
    Render a Chart.js-style config dict to PNG bytes using matplotlib.
    Used exclusively by _generate_pdf_reportlab to embed charts in PDF.
    Returns None on any failure (chart is silently skipped in that case).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io as _io
        import numpy as _np

        data       = chart_cfg.get("data", {})
        labels     = data.get("labels", [])
        datasets   = data.get("datasets", [])
        chart_type = chart_cfg.get("type", "bar")
        options    = chart_cfg.get("options", {})
        title_text = (
            options.get("plugins", {}).get("title", {}).get("text", "")
            or chart_cfg.get("title", "")
        )

        if not labels or not datasets:
            return None

        _C = ["#6366f1","#10b981","#f59e0b","#ef4444",
              "#3b82f6","#ec4899","#8b5cf6","#14b8a6","#f97316","#a855f7"]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f9fafb")

        if chart_type == "pie":
            vals = [float(v) if isinstance(v, (int, float)) else 0.0
                    for v in (datasets[0].get("data", []) if datasets else [])]
            ax.pie(vals, labels=labels, colors=_C[:len(vals)],
                   autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 8})
            ax.set_facecolor("white")

        elif chart_type == "line":
            x = _np.arange(len(labels))
            for i, ds in enumerate(datasets):
                vals = [float(v) if isinstance(v, (int, float)) else 0.0
                        for v in ds.get("data", [])]
                ax.plot(x, vals, marker="o", markersize=4,
                        color=_C[i % len(_C)],
                        label=ds.get("label", f"Series {i+1}"), linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)
            if len(datasets) > 1:
                ax.legend(fontsize=8)

        else:  # bar
            x    = _np.arange(len(labels))
            n    = len(datasets)
            w    = 0.7 / max(n, 1)
            for i, ds in enumerate(datasets):
                vals = [float(v) if isinstance(v, (int, float)) else 0.0
                        for v in ds.get("data", [])]
                ax.bar(x + (i - (n-1)/2) * w, vals, width=w * 0.9,
                       color=_C[i % len(_C)],
                       label=ds.get("label", f"Series {i+1}"), zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            if n > 1:
                ax.legend(fontsize=8)

        scales = options.get("scales", {})
        if scales.get("x", {}).get("title", {}).get("text"):
            ax.set_xlabel(scales["x"]["title"]["text"], fontsize=9)
        if scales.get("y", {}).get("title", {}).get("text"):
            ax.set_ylabel(scales["y"]["title"]["text"], fontsize=9)
        if title_text:
            ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8)

        ax.tick_params(axis="both", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout(pad=1.2)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return buf.getvalue()
    except Exception as _e:
        print(f"   ⚠️  _chart_to_png_pdf: {_e}")
        return None


def _generate_pdf_reportlab(title: str, content: str, source_docs: List[str] = None,
                            charts: Optional[List[Dict]] = None) -> bytes:
    """
    Convert a Markdown report into a polished PDF using ReportLab.

    Layout:
      Page 1  — cover (title + date + divider)
      Page 2+ — report content with running header/footer

    This function never raises — it returns at minimum a minimal valid PDF.
    """
    buf = BytesIO()
    styles = _build_styles()
    date_str = datetime.now().strftime("%B %d, %Y")

    # Page callback closure
    def _on_first_page(c, d):
        _on_page(c, d, title, date_str)

    def _on_later_pages(c, d):
        _on_page(c, d, title, date_str)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.8 * cm, bottomMargin=2.5 * cm,
        title=title, author="Bimlo Copilot",
    )

    story: List = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))

    # Company logo
    # Layout: Backend/services/report_agent.py -> Frontend/public/favicon.png
    # Go up: services/ -> Backend/ -> BIMLO/ -> then into Frontend/public/
    _SERVICES_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../BIMLO/Backend/services
    _BACKEND_DIR     = os.path.dirname(_SERVICES_DIR)                # .../BIMLO/Backend
    _BIMLO_DIR       = os.path.dirname(_BACKEND_DIR)                 # .../BIMLO
    _FRONTEND_PUBLIC = os.path.join(_BIMLO_DIR, "Frontend", "public")
    _LOGO_SVG        = os.path.join(_FRONTEND_PUBLIC, "favicon.svg")
    _LOGO_PNG        = os.path.join(_FRONTEND_PUBLIC, "favicon.png")
    _LOGO_PNG_CACHE = os.path.join(tempfile.gettempdir(), "bimlo_logo_cover.png")
    _logo_rendered = False

    def _try_render_logo(source_path: str) -> bool:
        """Convert source_path → safe RGBA PNG cache, then add to story. Returns True on success."""
        try:
            from PIL import Image as PILImage
            with PILImage.open(source_path) as im:
                im = im.convert("RGBA")
                im.save(_LOGO_PNG_CACHE, format="PNG")
            logo_img = Image(_LOGO_PNG_CACHE, width=2.2 * cm, height=2.2 * cm)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 0.4 * cm))
            return True
        except Exception as _e:
            print(f"   ℹ️  Logo Pillow render failed ({source_path}): {_e}")
            return False

    # 1. Try SVG → PNG via cairosvg (best quality)
    if not _logo_rendered and os.path.exists(_LOGO_SVG):
        try:
            import cairosvg
            cairosvg.svg2png(url=_LOGO_SVG, write_to=_LOGO_PNG_CACHE, output_width=120, output_height=120)
            logo_img = Image(_LOGO_PNG_CACHE, width=2.2 * cm, height=2.2 * cm)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 0.4 * cm))
            _logo_rendered = True
            print("   ✅ Logo: rendered via cairosvg")
        except Exception as _svg_err:
            print(f"   ℹ️  Logo SVG/cairosvg skipped: {_svg_err}")

    # 2. Fallback: PNG via Pillow (handles palette/ICO quirks ReportLab can't)
    if not _logo_rendered and os.path.exists(_LOGO_PNG):
        _logo_rendered = _try_render_logo(_LOGO_PNG)
        if _logo_rendered:
            print("   ✅ Logo: rendered via Pillow (PNG)")

    # 3. Last resort: feed PNG directly to ReportLab (works for clean RGBA/RGB PNGs)
    if not _logo_rendered and os.path.exists(_LOGO_PNG):
        try:
            logo_img = Image(_LOGO_PNG, width=2.2 * cm, height=2.2 * cm)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 0.4 * cm))
            _logo_rendered = True
            print("   ✅ Logo: rendered directly via ReportLab")
        except Exception as _direct_err:
            print(f"   ⚠️  Logo: all attempts failed — {_direct_err}")

    if not _logo_rendered:
        story.append(Spacer(1, 1 * cm))

    story.append(Paragraph(title, styles["cover_title"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="50%", thickness=2, color=_BRAND_BLUE,
                             hAlign="CENTER", spaceAfter=12))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated by Bimlo Copilot  ·  {date_str}",
                            styles["cover_meta"]))
    if source_docs:
        story.append(Spacer(1, 0.4 * cm))
        docs_text = "Sources: " + " · ".join(source_docs[:6])
        story.append(Paragraph(docs_text, styles["cover_meta"]))
    story.append(PageBreak())

    # ── Report content — with chart injection ────────────────────────────────
    # Build a chart_id → chart_config lookup for quick access
    charts_lookup: Dict[str, Dict] = {}
    if charts:
        for ch in charts:
            charts_lookup[ch["chart_id"]] = ch

    if charts_lookup:
        # Split content on <!-- CHART:id --> markers and interleave images
        import re as _re
        parts = _re.split(r'<!-- CHART:([a-zA-Z0-9_-]+) -->', content)
        # parts alternates: text, chart_id, text, chart_id, ...
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                # text segment
                story.extend(_md_to_story(part, styles))
            else:
                # chart_id
                ch = charts_lookup.get(part)
                if ch:
                    # Chart title
                    ch_title = ch.get("title", "")
                    if ch_title:
                        story.append(Spacer(1, 6))
                        story.append(Paragraph(f"<b>{ch_title}</b>", styles["h3"]))
                    # Render chart_js config → PNG
                    png_bytes = _chart_to_png_pdf(ch.get("chart_js", {}))
                    if png_bytes:
                        from io import BytesIO as _BIO
                        img_buf = _BIO(png_bytes)
                        chart_img = Image(img_buf, width=14 * cm, height=7.5 * cm)
                        chart_img.hAlign = "CENTER"
                        story.append(Spacer(1, 4))
                        story.append(chart_img)
                        story.append(Spacer(1, 4))
                    # Interpretation note
                    interp = ch.get("interpretation", "")
                    if interp:
                        story.append(Paragraph(f"<i>{interp}</i>", styles["blockquote"]))
    else:
        content_story = _md_to_story(content, styles)
        story.extend(content_story)

    # ── Sources footer ────────────────────────────────────────────────────────
    if source_docs:
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_GRAY_BORDER,
                                 spaceAfter=6))
        story.append(Paragraph("<b>Source Documents</b>", styles["source"]))
        for doc_name in source_docs:
            story.append(Paragraph(f"• {doc_name}", styles["source"]))

    try:
        doc.build(story,
                  onFirstPage=_on_first_page,
                  onLaterPages=_on_later_pages)
        pdf_bytes = buf.getvalue()
        print(f"   ✅ PDF: ReportLab OK ({len(pdf_bytes):,} bytes)")
        return pdf_bytes
    except Exception as e:
        print(f"   ⚠️  PDF: ReportLab error: {e} — falling back to minimal PDF")
        return _minimal_pdf_fallback(title, content)


def _minimal_pdf_fallback(title: str, content: str) -> bytes:
    """
    Ultra-safe fallback: plain ReportLab doc with no custom styling.
    Should only ever be reached if the full builder crashes on exotic input.
    """
    buf = BytesIO()
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
    ]
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
        elif stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            story.append(Paragraph(heading, styles["Heading1"]))
        else:
            # Strip any markdown markup for safety
            plain = re.sub(r'[*_`]', '', stripped)
            story.append(Paragraph(plain, styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


def _generate_md(content: str) -> str:
    """Strip <!-- CHART:id --> placeholders from Markdown (charts are PDF-only)."""
    import re as _re
    return _re.sub(r'\n*<!-- CHART:[a-zA-Z0-9_-]+ -->\n*', '\n\n', content).strip()


def _generate_pdf(title: str, content: str, source_docs: List[str] = None,
                  charts: Optional[List[Dict]] = None) -> bytes:
    """
    Public entry: always returns valid PDF bytes.
    Charts are rendered as embedded PNG images in the PDF only.
    """
    print("   PDF: rendering with ReportLab...")
    return _generate_pdf_reportlab(title, content, source_docs or [], charts or [])


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CONTEXT  (written by main.py after every /query call)
# ═══════════════════════════════════════════════════════════════════════════════

class SharedContext:
    """
    In-process store for per-session RAG state.
    Written by main.py after every /query; read by report_agent and other agents.
    """
    _history:               Dict[str, List[Dict]] = {}
    _chunks:                Dict[str, List[Dict]] = {}
    _analytics:             Dict[str, Any]        = {}
    _vector_store:          Any                   = None
    _pending_chart_clarif:  Dict[str, bool]       = {}  # True while awaiting chart clarification answer

    @classmethod
    def set_vector_store(cls, vs: Any) -> None:
        """Call once from main.py after VectorStoreManager is created."""
        cls._vector_store = vs

    @classmethod
    def set_history(cls, session_id: str, history: List[Dict]) -> None:
        cls._history[session_id] = history[-40:]

    @classmethod
    def get_history(cls, session_id: str) -> List[Dict]:
        return cls._history.get(session_id, [])

    @classmethod
    def set_chunks(cls, session_id: str, chunks: List[Dict]) -> None:
        cls._chunks[session_id] = chunks[:30]

    @classmethod
    def get_chunks(cls, session_id: str) -> List[Dict]:
        return cls._chunks.get(session_id, [])

    @classmethod
    def set_analytics(cls, session_id: str, analytics: Any) -> None:
        cls._analytics[session_id] = analytics

    @classmethod
    def get_analytics(cls, session_id: str) -> Any:
        return cls._analytics.get(session_id)

    @classmethod
    def set_pending_chart_clarif(cls, session_id: str, pending: bool) -> None:
        cls._pending_chart_clarif[session_id] = pending

    @classmethod
    def get_pending_chart_clarif(cls, session_id: str) -> bool:
        return cls._pending_chart_clarif.get(session_id, False)

    @classmethod
    def fetch_chunks_for_report(
        cls,
        prompt: str,
        available_docs: List[str],
        explicit_docs: List[str],
        top_k: int = 25,
    ) -> List[Dict]:
        """
        Direct vector-store retrieval used when SharedContext has no cached chunks
        for the session (e.g. the report was triggered before /query finished, or
        the user opened the report panel in a fresh session).

        Runs up to 3 query passes:
          1. Full prompt against all docs
          2. Doc-filtered pass for any explicitly mentioned files
          3. Broad sweep of each explicit doc to maximise coverage
        """
        vs = cls._vector_store
        if vs is None:
            print("   ⚠️  SharedContext.fetch_chunks_for_report: no vector_store injected")
            return []

        seen_ids: set = set()
        results:  List[Dict] = []

        def _add(chunks):
            for c in chunks:
                cid = c.get("id") or c.get("chunk_id") or str(c.get("metadata", {}))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(c)

        try:
            # Pass 1: semantic search on the full prompt
            _add(vs.search(prompt, top_k=top_k))

            # Pass 2: file-filtered search for explicitly requested docs
            for doc in (explicit_docs or [])[:5]:
                _add(vs.search(prompt, top_k=15, filter_dict={"filename": doc}))

            # Pass 3: if still thin, broad sweep of each explicit doc
            if len(results) < 8 and explicit_docs:
                for doc in explicit_docs[:3]:
                    doc_name = doc.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
                    _add(vs.search(doc_name, top_k=20, filter_dict={"filename": doc}))

            # Pass 4: if we have available_docs but nothing explicit, sweep the first few
            if len(results) < 5 and not explicit_docs and available_docs:
                for doc in available_docs[:3]:
                    _add(vs.search(prompt, top_k=10, filter_dict={"filename": doc}))

            print(f"   📂 SharedContext.fetch_chunks_for_report: {len(results)} chunks retrieved")
        except Exception as e:
            print(f"   ⚠️  SharedContext.fetch_chunks_for_report error: {e}")

        return results[:40]


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE  (in-memory + optional JSON file backing)
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_DIR    = os.getenv("DATA_DIR", "/home/claude/bimlo-copilot/data")
_REPORTS_DIR = os.path.join(_DATA_DIR, "reports")
os.makedirs(_REPORTS_DIR, exist_ok=True)

_reports_store: Dict[str, Dict] = {}


def _load_reports_from_disk() -> None:
    for p in Path(_REPORTS_DIR).glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "report_id" in data:
                _reports_store[data["report_id"]] = data
        except Exception:
            pass


def _save_report_to_disk(report: Dict) -> None:
    path = os.path.join(_REPORTS_DIR, f"{report['report_id']}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  report_agent: could not save {path}: {e}")


def _delete_report_from_disk(report_id: str) -> None:
    path = os.path.join(_REPORTS_DIR, f"{report_id}.json")
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


_load_reports_from_disk()


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerateReportRequest(BaseModel):
    prompt:         str
    session_id:     Optional[str] = None   # auto-generated if omitted
    available_docs: List[str] = []
    explicit_docs:  List[str] = []
    include_charts: bool = True
    language:       Optional[str] = None


class PatchReportRequest(BaseModel):
    instruction: str
    session_id:  Optional[str] = None
    language:    Optional[str] = None


class RestoreVersionRequest(BaseModel):
    version: int


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_context(chunks: List[Dict], max_chars: int = 12_000) -> str:
    parts: List[str] = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk.get("metadata", {})
        fname = meta.get("filename", "unknown")
        text  = chunk.get("text", "").strip()
        if not text:
            continue
        entry = f"[Source {i} | {fname}]\n{text}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(entry[:remaining] + "…")
            break
        parts.append(entry)
        total += len(entry) + 2
    return "\n\n".join(parts)


def _build_history_summary(history: List[Dict], max_turns: int = 10) -> str:
    if not history:
        return ""
    recent = history[-max_turns * 2:]
    lines  = []
    for msg in recent:
        role    = msg.get("role", "user").upper()
        content = msg.get("content", "")[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SMART REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_llm_json(raw: str) -> Optional[Dict]:
    """Robustly parse JSON from an LLM response (handles fences, single quotes, etc.)."""
    if not raw:
        return None
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Try standard JSON
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting first {...} block
    m = re.search(r'\{[\s\S]*\}', clean)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _generate_report_content(
    prompt: str,
    chunks: List[Dict],
    history: List[Dict],
    language: Optional[str],
    available_docs: List[str],
    explicit_docs: List[str],
    analytics: Any = None,
    requested_charts: Optional[List[Dict]] = None,  # pre-built chart configs from clarification flow
) -> Dict[str, Any]:
    """
    Generate a fully grounded, query-aware report.

    Steps:
      1. Extract user intent (what tables/charts/sections they want).
      2. Plan section headings driven by intent + source docs.
      3. Generate each section STRICTLY from source text.
      4. Optionally embed chart data from session analytics.
      5. Generate a short conversational summary for the chat reply.

    Returns: {title, content (Markdown), charts, source_docs, language, summary}
    """
    # ── Filter chunks to only the files the user asked about ─────────────────
    # If the user named specific files, restrict context to those files only.
    # The LLM prompt also reinforces this, but filtering here is the hard guard.
    if explicit_docs:
        explicit_lower = [d.lower() for d in explicit_docs]
        filtered = [
            c for c in chunks
            if any(
                ed in (c.get("metadata", {}).get("filename", "") or "").lower()
                for ed in explicit_lower
            )
        ]
        # Fall back to all chunks only if filtering wiped everything out
        # (shouldn't happen after fetch_chunks_for_report, but be safe)
        working_chunks = filtered if filtered else chunks
        print(f"   📎 explicit_docs filter: {len(chunks)} → {len(working_chunks)} chunks "
              f"(files: {explicit_docs})")
    else:
        working_chunks = chunks

    context      = _build_context(working_chunks, max_chars=16_000)
    history_text = _build_history_summary(history)
    docs_hint    = ", ".join(explicit_docs) if explicit_docs else ", ".join(available_docs[:8])

    # ── Step 0: Detect intent ─────────────────────────────────────────────────
    intent_system = (
        "You are a report planner. Extract exactly what the user wants in their report. "
        "Respond ONLY in valid JSON — no markdown fences, no preamble."
    )
    intent_prompt = (
        f'User request: "{prompt}"\n\n'
        'Return JSON:\n'
        '{\n'
        '  "wants_tables": true|false,\n'
        '  "wants_charts": true|false,\n'
        '  "wants_comparison": true|false,\n'
        '  "specific_topics": ["topic1", "topic2"],\n'
        '  "specific_files": ["filename.pdf"],\n'
        '  "report_type": "analysis|summary|technical_brief|comparison|audit|proposal",\n'
        '  "language": "en|fr|ar|..."\n'
        '}'
    )

    intent: Dict[str, Any] = {}
    raw_intent = call_llm(intent_prompt, system_prompt=intent_system,
                          max_tokens=300, temperature=0.1, task="plan")
    parsed_intent = _parse_llm_json(raw_intent)
    if parsed_intent:
        intent = parsed_intent

    detected_language = intent.get("language") or language or "en"
    wants_tables      = bool(intent.get("wants_tables", False))
    wants_charts      = bool(intent.get("wants_charts", False)) or (analytics is not None)
    specific_topics   = intent.get("specific_topics", [])
    report_type       = intent.get("report_type", "analysis")

    # ── Step 1: Plan sections ─────────────────────────────────────────────────
    plan_system = (
        "You are an expert analyst. Plan a professional report structure. "
        "Every planned section MUST be answerable from the provided document excerpts. "
        "Do NOT plan sections that require information not in the documents. "
        "Respond ONLY in valid JSON — no markdown fences, no preamble."
    )

    lang_directive = (
        f"The report MUST be written in {detected_language}."
        if detected_language != "en"
        else "Write the report in English."
    )
    topics_hint = f"The user specifically wants coverage of: {', '.join(specific_topics)}." if specific_topics else ""
    tables_hint = "Include data tables where relevant." if wants_tables else ""
    charts_hint = "Flag sections where a chart would add value (mark with needs_chart: true)." if wants_charts else ""
    files_hint  = (
        f"CRITICAL: The user explicitly asked for a report on ONLY these file(s): {', '.join(explicit_docs)}. "
        f"The source_docs field MUST contain only these files. Do NOT reference any other documents."
        if explicit_docs else ""
    )

    plan_prompt = (
        f'User request: "{prompt}"\n'
        f"{lang_directive}\n"
        f"{topics_hint}\n"
        f"{files_hint}\n"
        f"{tables_hint}\n"
        f"{charts_hint}\n\n"
        f"Source documents for this report: {docs_hint or 'the indexed documents'}\n\n"
        f"DOCUMENT EXCERPTS (sections must be grounded ONLY in this content):\n"
        f"{context[:6000] or '(no documents indexed yet)'}\n\n"
        f"Recent conversation:\n{history_text or '(none)'}\n\n"
        f"RULES:\n"
        f"- Plan 4-7 sections that best serve the request.\n"
        f"- Every section must be directly answerable from the excerpts above.\n"
        f"- Do NOT plan sections requiring info not present in those excerpts.\n"
        f"- source_docs must list ONLY the files actually used — {', '.join(explicit_docs) if explicit_docs else 'from the excerpts above'}.\n\n"
        f"TITLE RULES:\n"
        f"- NEVER repeat the user\'s request as the title. Generate an intelligent, editorial title.\n"
        f"- Must be specific: include the actual subject, entity, metric, or period from the documents.\n"
        f"- Good examples: \'Telecom Infrastructure Cost Breakdown 2024\', \'Customer Churn Drivers — Prepaid Segment\', \'Network KPI Benchmarking Report\'\n"
        f"- Bad examples: \'Report on revenue\', \'Analysis of the document\', \'Summary Report\'\n"
        f"- 4–9 words. Title Case. Do NOT start with \'Report\', \'Analysis\', or \'Summary\'.\n\n"
        f"Return ONLY this JSON:\n"
        f'{{\n'
        f'  "title": "...",\n'
        f'  "report_type": "{report_type}",\n'
        f'  "language": "{detected_language}",\n'
        f'  "sections": [\n'
        f'    {{"id": "s1", "heading": "...", "instruction": "what to cover, be specific", "needs_chart": false, "needs_table": false}}\n'
        f'  ],\n'
        f'  "source_docs": {json.dumps(explicit_docs) if explicit_docs else json.dumps(["filename1.pdf"])}\n'
        f'}}'
    )

    raw_plan = call_llm(plan_prompt, system_prompt=plan_system,
                        max_tokens=900, temperature=0.15, task="plan")
    plan = _parse_llm_json(raw_plan)

    if not plan or not isinstance(plan.get("sections"), list):
        # Graceful fallback plan
        plan = {
            "title":       f"Report: {prompt[:60]}",
            "report_type": report_type,
            "language":    detected_language,
            "sections": [
                {"id": "s1", "heading": "Overview",    "instruction": "Summarise key findings from the documents only.", "needs_chart": False, "needs_table": False},
                {"id": "s2", "heading": "Analysis",    "instruction": "Analyse main topics using only the provided sources.", "needs_chart": False, "needs_table": False},
                {"id": "s3", "heading": "Conclusions", "instruction": "Draw conclusions strictly from the documents.", "needs_chart": False, "needs_table": False},
            ],
            "source_docs": explicit_docs if explicit_docs else list({
                c.get("metadata", {}).get("filename", "")
                for c in working_chunks
                if c.get("metadata", {}).get("filename")
            })[:5],
        }

    raw_title = plan.get("title", "Report")
    # Refine the title with a dedicated focused LLM call — the planner often
    # echoes back the user's prompt; this single-purpose call produces a clean,
    # editorial title.
    _title_system = (
        "You generate short, specific report titles. "
        "Rules: 4–8 words, Title Case, NO quotes, NO punctuation at end. "
        "Must name the real subject/entity/metric from the documents. "
        "NEVER start with 'Report', 'Analysis', or 'Summary'. "
        "NEVER just reword the user's request. "
        "Good examples: 'Telecom Site Survey Field Results', "
        "'Network Infrastructure Cost Breakdown 2024', 'Customer Churn Drivers — Prepaid'. "
        "Reply with ONLY the title."
    )
    _title_prompt = (
        f"Report request: \"{prompt}\"\n"
        f"Document(s): {docs_hint or 'various'}\n"
        f"Planned title candidate: \"{raw_title}\"\n\n"
        "Generate a better, more specific title:"
    )
    _refined = call_llm(_title_prompt, system_prompt=_title_system,
                        max_tokens=30, temperature=0.2, task="classify").strip().strip('"').strip("'")
    title = _refined if (4 <= len(_refined) <= 100) else raw_title
    # Hard-enforce: if user named specific files, source_docs must only be those files
    source_docs = explicit_docs if explicit_docs else plan.get("source_docs", [])

    # ── Step 2: Generate each section ─────────────────────────────────────────
    section_system = (
        f"You are a technical writer producing a {report_type} in {detected_language}. "
        "CRITICAL RULE: Write ONLY what is directly supported by the provided document excerpts. "
        "If a fact is not in the excerpts, do NOT include it. "
        "Cite the source document name when you use specific data. "
        "Use markdown. Be specific: use exact numbers, names, and dates from the sources."
        + (f" You MUST only draw from these files: {', '.join(explicit_docs)}." if explicit_docs else "")
    )

    sections_md: List[str] = []

    for sec in plan.get("sections", []):
        heading     = sec.get("heading", "Section")
        instruction = sec.get("instruction", "Write this section.")
        needs_table = sec.get("needs_table", False) or wants_tables

        table_hint = (
            "\n\nIf the data supports it, present key figures as a Markdown table "
            "(use | column | headers | format)."
            if needs_table else ""
        )

        files_only_hint = (
            f"\n\nSOURCE RESTRICTION: Base this section ONLY on content from: {', '.join(explicit_docs)}. "
            "Do not reference or use content from any other document."
            if explicit_docs else ""
        )

        sec_prompt = (
            f'Report title: "{title}"\n'
            f"Report type: {report_type}\n"
            f"Language: {detected_language}\n"
            f'User original request: "{prompt}"\n\n'
            f'This section: "{heading}"\n'
            f"What to cover: {instruction}{table_hint}{files_only_hint}\n\n"
            f"DOCUMENT EXCERPTS — use ONLY these as your source of facts:\n"
            f"{context}\n\n"
            f"STRICT RULES:\n"
            f"1. Every claim must come directly from the document excerpts above.\n"
            f"2. Do NOT invent data, estimates, or context not present in the excerpts.\n"
            f"3. If information is sparse for a sub-point, note the gap briefly and move on — do NOT fill an entire section with 'The available documents do not specify...' repetitions.\n"
            f"4. Cite the source document name for specific facts.\n"
            f"5. Do NOT repeat the section heading — start directly with content.\n"
            f"6. Length: 2-4 focused paragraphs."
        )

        sec_content = call_llm(
            prompt=sec_prompt,
            system_prompt=section_system,
            max_tokens=800,
            temperature=0.2,
            task="synthesise",
        )

        sections_md.append(f"## {heading}\n\n{sec_content.strip()}")

    # ── Step 3: Build and embed charts ───────────────────────────────────────
    charts_for_report: List[Dict] = []

    # Helper: render a Chart.js-style data dict → PNG bytes via matplotlib
    def _chart_to_png(chart_cfg: Dict) -> Optional[bytes]:
        """Convert a chart_js config dict to PNG bytes using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io as _io
            import numpy as _np

            data = chart_cfg.get("data", {})
            labels = data.get("labels", [])
            datasets = data.get("datasets", [])
            chart_type = chart_cfg.get("type", "bar")
            options = chart_cfg.get("options", {})
            title_text = (
                options.get("plugins", {}).get("title", {}).get("text", "")
                or chart_cfg.get("title", "")
            )

            if not labels or not datasets:
                return None

            fig, ax = plt.subplots(figsize=(7, 3.8))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#f9fafb")

            _COLORS = ["#6366f1","#10b981","#f59e0b","#ef4444",
                       "#3b82f6","#ec4899","#8b5cf6","#14b8a6",
                       "#f97316","#a855f7"]

            if chart_type == "pie":
                raw_data = datasets[0].get("data", []) if datasets else []
                vals = [float(v) if isinstance(v, (int, float)) else 0.0 for v in raw_data]
                colors = _COLORS[:len(vals)]
                wedges, texts, autotexts = ax.pie(
                    vals, labels=labels, colors=colors,
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 8},
                )
                for at in autotexts:
                    at.set_fontsize(7)
                ax.set_facecolor("white")

            elif chart_type == "line":
                x = _np.arange(len(labels))
                for i, ds in enumerate(datasets):
                    vals = [float(v) if isinstance(v, (int, float)) else 0.0 for v in ds.get("data", [])]
                    ax.plot(x, vals, marker="o", markersize=4,
                            color=_COLORS[i % len(_COLORS)],
                            label=ds.get("label", f"Series {i+1}"), linewidth=2)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                ax.set_axisbelow(True)
                if len(datasets) > 1:
                    ax.legend(fontsize=8, framealpha=0.8)

            else:  # bar (default)
                x = _np.arange(len(labels))
                n = len(datasets)
                width = 0.7 / max(n, 1)
                for i, ds in enumerate(datasets):
                    vals = [float(v) if isinstance(v, (int, float)) else 0.0 for v in ds.get("data", [])]
                    offset = (i - (n - 1) / 2) * width
                    ax.bar(x + offset, vals, width=width * 0.9,
                           color=_COLORS[i % len(_COLORS)],
                           label=ds.get("label", f"Series {i+1}"), zorder=3)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
                ax.set_axisbelow(True)
                if n > 1:
                    ax.legend(fontsize=8, framealpha=0.8)

            # Axis labels from options
            scales = options.get("scales", {})
            x_title = scales.get("x", {}).get("title", {}).get("text", "")
            y_title = scales.get("y", {}).get("title", {}).get("text", "")
            if x_title:
                ax.set_xlabel(x_title, fontsize=9)
            if y_title:
                ax.set_ylabel(y_title, fontsize=9)

            if title_text:
                ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8)

            ax.tick_params(axis="both", labelsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            plt.tight_layout(pad=1.2)
            buf_img = _io.BytesIO()
            fig.savefig(buf_img, format="png", dpi=130, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            return buf_img.getvalue()
        except Exception as _e:
            print(f"   ⚠️  _chart_to_png: {_e}")
            return None

    # Helper: add a chart to the report (charts_for_report list + markdown placeholder)
    def _embed_chart(chart_result: Dict, section_id: str) -> Optional[str]:
        """
        Given a build_chart() result dict, register it and return the
        markdown placeholder string to splice into section content, or None.
        """
        if chart_result.get("type") != "chart_config":
            return None
        chart_id = str(uuid.uuid4())
        charts_for_report.append({
            "section_id":     section_id,
            "chart_id":       chart_id,
            "chart_js":       chart_result.get("chart_js", {}),
            "title":          chart_result.get("title", "Chart"),
            "description":    chart_result.get("description", ""),
            "interpretation": chart_result.get("interpretation", ""),
        })
        return f"\n\n<!-- CHART:{chart_id} -->\n"

    # 3a. Embed any pre-built charts from the clarification flow
    if requested_charts:
        for rc in requested_charts:
            _embed_chart(rc, "user_requested")

    # 3b. Build charts for sections the planner flagged as needs_chart
    # (only when no pre-built charts were passed in for those sections)
    _graph_agent_mod = None
    try:
        import importlib as _il
        import sys as _sys
        if "services.graph_agent" in _sys.modules:
            _graph_agent_mod = _sys.modules["services.graph_agent"]
        else:
            try:
                _graph_agent_mod = _il.import_module("services.graph_agent")
            except ModuleNotFoundError:
                _graph_agent_mod = _il.import_module("graph_agent")
    except Exception:
        pass

    chart_sections_md: List[str] = []
    for i, (sec, sec_md) in enumerate(zip(plan.get("sections", []), sections_md)):
        if sec.get("needs_chart") and _graph_agent_mod:
            try:
                _ga = _graph_agent_mod.GraphAgent()
                chart_result = _ga.build_chart(
                    query=f"{sec.get('instruction', sec.get('heading', ''))} from {docs_hint}",
                    chunks=working_chunks,
                    language=detected_language,
                    skip_clarification=True,  # planner already decided; no looping
                )
                placeholder = _embed_chart(chart_result, sec.get("id", f"s{i}"))
                if placeholder:
                    sec_md = sec_md + placeholder
                    print(f"   📊 Chart embedded in section '{sec.get('heading')}'")
            except Exception as _ce:
                print(f"   ⚠️  Chart for section '{sec.get('heading')}': {_ce}")
        chart_sections_md.append(sec_md)

    sections_md = chart_sections_md if chart_sections_md else sections_md

    # 3c. Embed analytics chart from session if available (legacy path)
    if analytics and isinstance(analytics, dict):
        chart_cfg = analytics.get("chart_config") or analytics
        if chart_cfg and isinstance(chart_cfg, dict) and chart_cfg.get("type") == "chart_config":
            placeholder = _embed_chart(chart_cfg, "analytics")
            if placeholder:
                sections_md.append(
                    f"## Data Visualisation\n\n"
                    f"The following chart was generated from the data in your documents."
                    + placeholder
                )

    full_content = f"# {title}\n\n" + "\n\n".join(sections_md)

    # ── Step 4: Brief conversational summary ─────────────────────────────────
    # Feed the actual first ~1200 chars of content so the LLM can reference
    # real findings instead of generating hollow filler phrases.
    content_preview = full_content[:1200].strip()
    sections_list   = ", ".join(s.get("heading", "") for s in plan.get("sections", []))
    docs_phrase     = (
        f"{source_docs[0]}" if len(source_docs) == 1
        else " and ".join(source_docs) if len(source_docs) == 2
        else ", ".join(source_docs[:-1]) + f", and {source_docs[-1]}"
    ) if source_docs else "your documents"

    summary_system = (
        "You are a sharp, friendly assistant delivering a report to its requester. "
        "You write like a knowledgeable colleague handing over a finished piece of work — "
        "direct, warm, specific. No corporate fluff, no hollow filler. "
        "Never use phrases like 'valuable insights', 'comprehensive examination', "
        "'notable findings', 'delve into', or 'it is worth noting'. "
        "Never start with 'I have generated' or 'The report contains'. "
        "Start with 'Here's' or a direct reference to what's inside."
    )

    summary_prompt = (
        f'The user asked: "{prompt}"\n\n'
        f"The report titled \"{title}\" was just built from: {docs_phrase}.\n"
        f"Sections covered: {sections_list}.\n"
        f"Charts included: {'yes' if charts_for_report else 'no'}.\n\n"
        f"Here is the opening content of the report:\n\"\"\"\n{content_preview}\n\"\"\"\n\n"
        "Write 2-3 sentences in a natural, conversational tone that:\n"
        "1. Opens with 'Here's your report on ...' or similar — name the actual topic.\n"
        "2. Mentions 1-2 specific findings or themes pulled directly from the content above — be concrete, not vague.\n"
        "3. If only one source was used, say so naturally. If multiple, name them.\n"
        "4. Optionally note the sections if it adds useful context (e.g. 'covering X, Y, and Z').\n"
        "No bullet points. No preamble. Output only the 2-3 sentences."
    )

    summary = call_llm(
        prompt=summary_prompt,
        system_prompt=summary_system,
        max_tokens=180,
        temperature=0.5,
        task="synthesise",
    )

    return {
        "title":       title,
        "content":     full_content,
        "charts":      charts_for_report,
        "source_docs": source_docs,
        "language":    detected_language,
        "summary":     summary.strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(tags=["reports"])


@router.post("/report-stream")
async def report_stream(req: GenerateReportRequest):
    """
    Streaming SSE version of report generation.

    Emits the same event shape as /query-stream so the frontend can use
    the exact same runStreamingQuery path — no RAG call ever happens:
      { "type": "status",  "node": "...", "icon": "...", "message": "..." }
      { "type": "result",  "answer": "...", "sources": [...], "confidence": 1.0,
                           "report_id": "...", "session_id": "..." }
      { "type": "error",   "message": "..." }
    """
    import asyncio as _asyncio

    q: queue.Queue = queue.Queue()
    DONE = object()

    def _emit(node: str, icon: str, message: str):
        q.put({"type": "status", "node": node, "icon": icon, "message": message})

    def _run():
        try:
            _emit("report_intent",   "📋", "Detected report request…")

            session_id = req.session_id or str(uuid.uuid4())
            chunks     = SharedContext.get_chunks(session_id)
            history    = SharedContext.get_history(session_id)
            analytics  = SharedContext.get_analytics(session_id)

            if not chunks:
                _emit("report_retrieve", "📂", "Fetching document chunks…")
                chunks = SharedContext.fetch_chunks_for_report(
                    prompt         = req.prompt,
                    available_docs = req.available_docs,
                    explicit_docs  = req.explicit_docs,
                )
                if chunks:
                    SharedContext.set_chunks(session_id, chunks)
                    _emit("report_retrieve", "✅", f"Retrieved {len(chunks)} chunks from your documents")
                else:
                    _emit("report_retrieve", "⚠️", "No cached chunks — report may be sparse")

            _emit("report_plan",     "🗂️",  "Planning report structure…")
            _emit("report_write",    "✍️",  "Writing sections from your documents…")

            result = _generate_report_content(
                prompt         = req.prompt,
                chunks         = chunks,
                history        = history,
                language       = req.language,
                available_docs = req.available_docs,
                explicit_docs  = req.explicit_docs,
                analytics      = analytics if req.include_charts else None,
            )

            _emit("report_save", "💾", "Saving report…")

            now       = datetime.now().isoformat()
            report_id = str(uuid.uuid4())
            report: Dict[str, Any] = {
                "report_id":   report_id,
                "title":       result["title"],
                "content":     result["content"],
                "charts":      result.get("charts", []),
                "source_docs": result["source_docs"],
                "language":    result["language"],
                "summary":     result.get("summary", ""),
                "created_at":  now,
                "updated_at":  now,
                "version":     1,
                "versions": [{
                    "version":     1,
                    "title":       result["title"],
                    "instruction": req.prompt,
                    "created_at":  now,
                    "content":     result["content"],
                }],
                "session_id":  session_id,
            }
            _reports_store[report_id] = report
            _save_report_to_disk(report)
            print(f"   ✅ report-stream: saved {report_id!r} ({result['title']!r})")

            summary = result.get("summary", "").strip() or (
                f'The report "{result["title"]}" has been generated.'
            )

            word_count    = len(result["content"].split())
            section_count = len([l for l in result["content"].splitlines()
                                  if l.startswith("# ") or l.startswith("## ")])

            q.put({
                "type":          "result",
                "answer":        summary,
                "raw_answer":    summary,
                "sources":       [{"filename": d, "content": ""} for d in result["source_docs"]],
                "confidence":    1.0,
                "report_id":     report_id,
                "report_title":  result["title"],
                "report_meta": {
                    "word_count":    word_count,
                    "section_count": section_count,
                    "source_docs":   result["source_docs"],
                    "version":       1,
                },
                "session_id":    session_id,
            })

        except Exception as exc:
            import traceback; traceback.print_exc()
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(DONE)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _events():
        loop = _asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=120))
            except Exception:
                break
            if item is DONE:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/reports")
async def create_report(req: GenerateReportRequest):
    """
    Generate a new report grounded in the session's retrieved chunks.
    Returns the full report dict; the `summary` field is what the assistant
    message shows in chat.
    """
    print(f"\n📝 ReportAgent → session={req.session_id!r} prompt={req.prompt[:60]!r}")

    session_id = req.session_id or str(uuid.uuid4())
    chunks    = SharedContext.get_chunks(session_id)
    history   = SharedContext.get_history(session_id)
    analytics = SharedContext.get_analytics(session_id)

    if not chunks:
        print("   ⚠️  ReportAgent: no cached chunks — running direct retrieval fallback")
        chunks = SharedContext.fetch_chunks_for_report(
            prompt         = req.prompt,
            available_docs = req.available_docs,
            explicit_docs  = req.explicit_docs,
        )
        if chunks:
            SharedContext.set_chunks(session_id, chunks)
        else:
            print("   ⚠️  ReportAgent: fallback retrieval also empty — report may be sparse")

    try:
        result = _generate_report_content(
            prompt         = req.prompt,
            chunks         = chunks,
            history        = history,
            language       = req.language,
            available_docs = req.available_docs,
            explicit_docs  = req.explicit_docs,
            analytics      = analytics if req.include_charts else None,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Report generation failed: {e}")

    now       = datetime.now().isoformat()
    report_id = str(uuid.uuid4())

    report: Dict[str, Any] = {
        "report_id":   report_id,
        "title":       result["title"],
        "content":     result["content"],
        "charts":      result.get("charts", []),
        "source_docs": result["source_docs"],
        "language":    result["language"],
        "summary":     result.get("summary", ""),
        "created_at":  now,
        "updated_at":  now,
        "version":     1,
        "versions": [{
            "version":     1,
            "title":       result["title"],
            "instruction": req.prompt,
            "created_at":  now,
            "content":     result["content"],
        }],
        "session_id":  session_id,
    }

    _reports_store[report_id] = report
    _save_report_to_disk(report)
    print(f"   ✅ ReportAgent: saved → {report_id} ({result['title']!r})")

    return report


@router.get("/reports")
async def list_reports():
    """List all report summaries (no content body to keep payload small)."""
    summaries = []
    for r in sorted(_reports_store.values(), key=lambda x: x.get("updated_at", ""), reverse=True):
        summaries.append({
            "report_id":   r["report_id"],
            "title":       r["title"],
            "source_docs": r.get("source_docs", []),
            "created_at":  r.get("created_at"),
            "updated_at":  r.get("updated_at"),
            "version":     r.get("version", 1),
            "versions":    r.get("versions", []),
            "preview":     r["content"][:200].replace("\n", " ").strip(),
        })
    return summaries


@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    if report_id not in _reports_store:
        raise HTTPException(404, f"Report {report_id} not found")
    return _reports_store[report_id]


@router.patch("/reports/{report_id}")
async def patch_report(report_id: str, req: PatchReportRequest):
    """Edit a report with a natural-language instruction. Bumps version."""
    if report_id not in _reports_store:
        raise HTTPException(404, f"Report {report_id} not found")

    report  = _reports_store[report_id]
    sid     = req.session_id or report.get("session_id") or ""
    chunks  = SharedContext.get_chunks(sid)
    context = _build_context(chunks)

    system = (
        "You are an expert technical writer. "
        "You receive an existing report in Markdown and an edit instruction. "
        "Apply the instruction precisely and return ONLY the full updated Markdown. "
        "Preserve all section headings and overall structure unless the instruction explicitly says to change them."
    )

    prompt = (
        f'Existing report:\n"""\n{report["content"]}\n"""\n\n'
        f'Edit instruction: "{req.instruction}"\n\n'
        f"Language: {req.language or report.get('language', 'same as the report')}\n\n"
        f"Additional source context (use if relevant):\n{context[:4000] or '(none)'}\n\n"
        "Return ONLY the full updated Markdown report. No preamble or explanation."
    )

    updated_content = call_llm(
        prompt=prompt, system_prompt=system,
        max_tokens=3000, temperature=0.3, task="synthesise",
    )

    if not updated_content.strip():
        raise HTTPException(500, "LLM returned empty content for the edit")

    now         = datetime.now().isoformat()
    new_version = report["version"] + 1
    versions    = list(report.get("versions", []))
    versions.append({
        "version":     new_version,
        "title":       report["title"],
        "instruction": req.instruction,
        "created_at":  now,
        "content":     updated_content,
    })

    report = {
        **report,
        "content":    updated_content,
        "updated_at": now,
        "version":    new_version,
        "versions":   versions,
    }

    _reports_store[report_id] = report
    _save_report_to_disk(report)
    return report


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    if report_id not in _reports_store:
        raise HTTPException(404, f"Report {report_id} not found")
    del _reports_store[report_id]
    _delete_report_from_disk(report_id)
    return {"status": "deleted", "report_id": report_id}


@router.get("/reports/{report_id}/download")
async def download_report(report_id: str, fmt: str = "pdf"):
    """
    Download as:
    • fmt=md   → Markdown file
    • fmt=pdf  → ReportLab-generated PDF (always works, never empty)
    """
    if report_id not in _reports_store:
        raise HTTPException(404, f"Report {report_id} not found")

    report    = _reports_store[report_id]
    title     = report["title"]
    content   = report["content"]
    source_docs = report.get("source_docs", [])
    safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(" ", "_")[:50] or "report"

    if fmt == "md":
        md_content = _generate_md(content)
        return Response(
            content    = md_content.encode("utf-8"),
            media_type = "text/markdown; charset=utf-8",
            headers    = {"Content-Disposition": f'attachment; filename="{safe_name}.md"'},
        )

    # PDF — charts embedded as images
    charts = report.get("charts", [])
    pdf_bytes = _generate_pdf(title, content, source_docs, charts)
    return Response(
        content    = pdf_bytes,
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )


@router.post("/reports/{report_id}/restore")
async def restore_version(report_id: str, req: RestoreVersionRequest):
    """Restore a specific historical version."""
    if report_id not in _reports_store:
        raise HTTPException(404, f"Report {report_id} not found")

    report   = _reports_store[report_id]
    versions = report.get("versions", [])
    target   = next((v for v in versions if v.get("version") == req.version), None)

    if not target:
        raise HTTPException(404, f"Version {req.version} not found in report {report_id}")

    now         = datetime.now().isoformat()
    new_version = report["version"] + 1
    versions.append({
        "version":     new_version,
        "title":       target["title"],
        "instruction": f"Restored from v{req.version}",
        "created_at":  now,
        "content":     target.get("content", report["content"]),
    })

    report = {
        **report,
        "content":    target.get("content", report["content"]),
        "title":      target["title"],
        "updated_at": now,
        "version":    new_version,
        "versions":   versions,
    }

    _reports_store[report_id] = report
    _save_report_to_disk(report)
    return report