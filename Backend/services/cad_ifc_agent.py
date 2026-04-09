"""
cad_ifc_agent.py
────────────────────────────────────────────────────────────────────────────────
Unified CAD + IFC Analysis Agent for BIMLO Copilot.

Completely isolated from the RAG / telecom document pipeline.
Plugs into main.py exactly like news_chat_agent.py does.

Supported formats
─────────────────
  IFC  →  .ifc, .ifczip          (full BIM intelligence via IfcOpenShell)
  CAD  →  .dxf                   (geometry + layers via ezdxf)
  CAD  →  .dwg                   (ezdxf r2018 mode, best-effort)
  CAD  →  .step / .stp           (entity count via text scan — no OCC dep)

Pipeline
─────────
  0  Entry / Router       detect file type → branch IFC or CAD
  1  File Handling        validate, store temp, detect schema / 2D-3D
  2  Smart Routing        set capability flags + UX hint message
  3A IFC Parse            IfcOpenShell → elements, properties, relationships
  4A Normalization        IFC classes → clean labels, unify props, handle gaps
  5A Structuring          clean JSON summary
  6A Text embedding prep  element → natural-language strings for RAG (future)
  3B CAD Parse            ezdxf / text scan → geometry counts, layers
  4B CAD Structure        simplified JSON
  7  AI Reasoning         call_llm with structured context
  8  LLM Judge            score answer quality, retry once if poor
  9  SharedContext sync   same pattern as report_agent / news_chat_agent
 10  Logging              unknown types, missing props, parse errors

Endpoints
─────────
  POST  /api/cad/upload          upload + parse file → returns file_id + summary
  POST  /api/cad/query           ask a question about a previously uploaded file
  GET   /api/cad/files           list cached file summaries
  DELETE /api/cad/files/{fid}    remove from cache

Context sharing
───────────────
  CadSharedContext  mirrors SharedContext in report_agent.py:
    .set_summary(session_id, summary)
    .get_summary(session_id) → dict | None
  The /query endpoint reads this so session history is preserved.
"""

from __future__ import annotations

import io
import os
import re
import json
import uuid
import zipfile
import logging
import tempfile
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import queue

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# Internal CAD → IFC conversion pipeline (transparent to user)
try:
    from cad_to_ifc import convert_cad_to_ifc
    _CAD_TO_IFC_AVAILABLE = True
except ImportError:
    _CAD_TO_IFC_AVAILABLE = False
    logger.warning("[cad_ifc_agent] cad_to_ifc module not found — silent conversion disabled")

logger = logging.getLogger("cad_ifc_agent")
router = APIRouter()

# ── Context bridge — wires CAD turns into the shared session memory ──────────
try:
    from cad_context_bridge import sync_cad_turn_to_main as _sync_to_main
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False
    logger.warning("[cad_ifc_agent] cad_context_bridge not found — CAD turns won't sync to main session")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONTEXT  (mirrors report_agent.SharedContext)
# ─────────────────────────────────────────────────────────────────────────────

class CadSharedContext:
    """Thread-safe per-session context store for the CAD/IFC agent."""
    _lock = threading.Lock()
    _summaries:  Dict[str, dict]       = {}   # session_id → parsed summary
    _histories:  Dict[str, deque]      = {}   # session_id → conversation turns
    _file_cache: Dict[str, dict]       = {}   # file_id    → full parsed result

    MAX_HISTORY = 20

    # ── summary ──────────────────────────────────────────────────────────────
    @classmethod
    def set_summary(cls, session_id: str, summary: dict):
        with cls._lock:
            cls._summaries[session_id] = summary

    @classmethod
    def get_summary(cls, session_id: str) -> Optional[dict]:
        with cls._lock:
            return cls._summaries.get(session_id)

    # ── conversation history ──────────────────────────────────────────────────
    @classmethod
    def append_turn(cls, session_id: str, role: str, content: str):
        with cls._lock:
            if session_id not in cls._histories:
                cls._histories[session_id] = deque(maxlen=cls.MAX_HISTORY)
            cls._histories[session_id].append({"role": role, "content": content})

    @classmethod
    def get_history(cls, session_id: str) -> List[dict]:
        with cls._lock:
            return list(cls._histories.get(session_id, []))

    # ── file cache ────────────────────────────────────────────────────────────
    @classmethod
    def cache_file(cls, file_id: str, result: dict):
        with cls._lock:
            cls._file_cache[file_id] = result

    @classmethod
    def get_file(cls, file_id: str) -> Optional[dict]:
        with cls._lock:
            return cls._file_cache.get(file_id)

    @classmethod
    def list_files(cls) -> List[dict]:
        with cls._lock:
            return [
                {
                    "file_id":   fid,
                    "filename":  v.get("filename", ""),
                    "file_type": v.get("file_type", ""),
                    "pipeline":  v.get("pipeline", ""),
                    "cached_at": v.get("cached_at", ""),
                }
                for fid, v in cls._file_cache.items()
            ]

    @classmethod
    def delete_file(cls, file_id: str):
        with cls._lock:
            cls._file_cache.pop(file_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Supported extensions
IFC_EXTS = {".ifc", ".ifczip"}
CAD_EXTS = {".dwg", ".dxf", ".step", ".stp"}
ALL_EXTS  = IFC_EXTS | CAD_EXTS

# IFC class → human-readable label
_IFC_CLASS_LABELS: Dict[str, str] = {
    "IfcWall":              "wall",
    "IfcWallStandardCase": "wall",
    "IfcSlab":              "slab",
    "IfcBeam":              "beam",
    "IfcColumn":            "column",
    "IfcDoor":              "door",
    "IfcWindow":            "window",
    "IfcStair":             "stair",
    "IfcRoof":              "roof",
    "IfcSpace":             "space",
    "IfcBuildingStorey":    "storey",
    "IfcBuilding":          "building",
    "IfcSite":              "site",
    "IfcPipe":              "pipe",
    "IfcDuctSegment":       "duct",
    "IfcCableSegment":      "cable",
    "IfcFlowTerminal":      "flow terminal",
    "IfcFurnishingElement": "furniture",
    "IfcRailing":           "railing",
    "IfcCurtainWall":       "curtain wall",
    "IfcPlate":             "plate",
    "IfcMember":            "member",
    "IfcFooting":           "footing",
    "IfcPile":              "pile",
    "IfcDistributionElement": "distribution element",
}

def _ifc_label(ifc_class: str) -> str:
    """Map raw IfcClass → readable label; fallback to cleaned class name."""
    if ifc_class in _IFC_CLASS_LABELS:
        return _IFC_CLASS_LABELS[ifc_class]
    # Remove 'Ifc' prefix and split CamelCase
    clean = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', ifc_class.replace("Ifc", ""))
    return clean.lower().strip()


def _safe_get(obj, *attrs, default=None):
    """Safely traverse attribute chain; return default on any failure."""
    for attr in attrs:
        try:
            obj = getattr(obj, attr, None)
            if obj is None:
                return default
        except Exception:
            return default
    return obj if obj is not None else default


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FILE HANDLING LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _detect_type(filename: str) -> Tuple[str, str]:
    """
    Returns (pipeline, ext): pipeline ∈ {'ifc', 'cad', 'unsupported'}
    """
    ext = Path(filename).suffix.lower()
    if ext in IFC_EXTS:
        return "ifc", ext
    if ext in CAD_EXTS:
        return "cad", ext
    return "unsupported", ext


def _save_temp(data: bytes, suffix: str) -> str:
    """Write bytes to a named temp file, return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3A — IFC PARSING LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ifc(file_bytes: bytes, ext: str) -> dict:
    """
    Parse IFC file using IfcOpenShell.
    Returns normalized summary dict.
    """
    try:
        import ifcopenshell
    except ImportError:
        raise RuntimeError("ifcopenshell not installed — run: pip install ifcopenshell")

    # Handle .ifczip
    actual_bytes = file_bytes
    if ext == ".ifczip":
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            ifc_names = [n for n in zf.namelist() if n.lower().endswith(".ifc")]
            if not ifc_names:
                raise ValueError("No .ifc file found inside .ifczip")
            actual_bytes = zf.read(ifc_names[0])

    # Write to temp and open
    tmp = _save_temp(actual_bytes, ".ifc")
    try:
        model = ifcopenshell.open(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    # ── Detect schema ────────────────────────────────────────────────────────
    schema = getattr(model, "schema", "unknown").upper()   # e.g. "IFC4", "IFC2X3"

    # ── Extract elements ─────────────────────────────────────────────────────
    elements_raw: Dict[str, List[dict]] = {}   # label → [element dicts]
    unknown_classes: set = set()
    missing_props_log: list = []

    for entity in model.by_type("IfcProduct"):
        ifc_class = entity.is_a()
        label     = _ifc_label(ifc_class)

        if ifc_class not in _IFC_CLASS_LABELS:
            unknown_classes.add(ifc_class)

        # ── Properties ───────────────────────────────────────────────────────
        props: Dict[str, Any] = {}

        # GlobalId, Name
        gid  = _safe_get(entity, "GlobalId", default="")
        name = _safe_get(entity, "Name",     default="")
        if name:
            props["name"] = str(name)

        # Material
        try:
            rels = model.get_inverse(entity)
            for rel in rels:
                if rel.is_a("IfcRelAssociatesMaterial"):
                    mat = _safe_get(rel, "RelatingMaterial")
                    if mat:
                        mat_name = (
                            _safe_get(mat, "Name")
                            or _safe_get(mat, "ForLayerSet", "LayerSetName")
                            or mat.is_a()
                        )
                        if mat_name:
                            props["material"] = str(mat_name)
                    break
        except Exception:
            pass

        # Level (IfcBuildingStorey)
        try:
            for rel in model.get_inverse(entity):
                if rel.is_a("IfcRelContainedInSpatialStructure"):
                    container = _safe_get(rel, "RelatingStructure")
                    if container and container.is_a("IfcBuildingStorey"):
                        storey_name = _safe_get(container, "Name") or _safe_get(container, "LongName")
                        if storey_name:
                            props["level"] = str(storey_name)
                    break
        except Exception:
            pass

        # Pset_ property sets (height, width, length, thickness, etc.)
        KEY_PROPS = {"height", "width", "length", "thickness", "area", "volume",
                     "nominalheight", "nominalwidth", "nominallength",
                     "overallheight", "overallwidth", "isexternal"}
        try:
            for rel in model.get_inverse(entity):
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = _safe_get(rel, "RelatingPropertyDefinition")
                    if pset and pset.is_a("IfcPropertySet"):
                        for prop in (pset.HasProperties or []):
                            pname = (prop.Name or "").lower().replace(" ", "")
                            if pname in KEY_PROPS:
                                val = _safe_get(prop, "NominalValue", "wrappedValue")
                                if val is not None:
                                    props[pname] = val
        except Exception:
            pass

        # Flag if material or level missing (for logging)
        if "material" not in props:
            missing_props_log.append({"id": gid, "class": ifc_class, "missing": "material"})
        if "level" not in props and ifc_class not in ("IfcSite", "IfcBuilding", "IfcBuildingStorey"):
            missing_props_log.append({"id": gid, "class": ifc_class, "missing": "level"})

        elem_dict = {"id": gid, "type": label, "ifc_class": ifc_class, **props}

        if label not in elements_raw:
            elements_raw[label] = []
        elements_raw[label].append(elem_dict)

    # ── Storeys ───────────────────────────────────────────────────────────────
    storeys = []
    for s in model.by_type("IfcBuildingStorey"):
        sname = _safe_get(s, "Name") or _safe_get(s, "LongName") or "Unnamed"
        elev  = _safe_get(s, "Elevation")
        storeys.append({"name": str(sname), "elevation": elev})

    # ── Summary counts ────────────────────────────────────────────────────────
    element_counts = {label: len(items) for label, items in elements_raw.items()}
    total_elements = sum(element_counts.values())

    # ── Material inventory ────────────────────────────────────────────────────
    material_inventory: Dict[str, int] = {}
    for items in elements_raw.values():
        for el in items:
            mat = el.get("material")
            if mat:
                material_inventory[mat] = material_inventory.get(mat, 0) + 1

    # ── Embedding strings (for future RAG) ───────────────────────────────────
    embed_strings: List[str] = []
    for label, items in elements_raw.items():
        for el in items[:50]:   # cap to avoid huge payloads
            parts = [f"{el['type']}"]
            if el.get("material"): parts.append(f"material: {el['material']}")
            if el.get("level"):    parts.append(f"level: {el['level']}")
            if el.get("height"):   parts.append(f"height: {el['height']}")
            embed_strings.append(", ".join(parts))

    # ── Unknown / missing logs ────────────────────────────────────────────────
    if unknown_classes:
        logger.warning(f"[ifc_parse] Unknown IFC classes (not in label map): {unknown_classes}")
    if missing_props_log:
        logger.debug(f"[ifc_parse] {len(missing_props_log)} elements missing material/level")

    return {
        "pipeline":          "ifc",
        "schema":            schema,
        "total_elements":    total_elements,
        "element_counts":    element_counts,
        "storeys":           storeys,
        "material_inventory": material_inventory,
        "elements_sample":   {
            lbl: items[:5] for lbl, items in elements_raw.items()
        },
        "embed_strings":     embed_strings[:200],
        "unknown_classes":   list(unknown_classes),
        "missing_props_count": len(missing_props_log),
        "capability":        "full",
        "ux_hint":           "🧠 Advanced BIM analysis enabled",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3B — CAD PARSING LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cad_dimension(dxf_doc) -> str:
    """Heuristic: if any entity has non-zero Z, likely 3D."""
    try:
        msp = dxf_doc.modelspace()
        for entity in list(msp)[:500]:
            # check insert point or start/end point
            for attr in ("dxf.insert", "dxf.start", "dxf.center"):
                try:
                    parts = attr.split(".")
                    pt = getattr(getattr(entity, parts[0]), parts[1])
                    if hasattr(pt, "z") and abs(pt.z) > 1e-6:
                        return "3D"
                except Exception:
                    pass
    except Exception:
        pass
    return "2D"


def _parse_dxf(file_bytes: bytes) -> dict:
    """Parse DXF file with ezdxf."""
    try:
        import ezdxf
        from ezdxf.enums import UnitsCode
    except ImportError:
        raise RuntimeError("ezdxf not installed — run: pip install ezdxf")

    tmp = _save_temp(file_bytes, ".dxf")
    try:
        doc = ezdxf.readfile(tmp)
    except Exception as e:
        raise ValueError(f"DXF parse error: {e}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    msp = doc.modelspace()

    # Count entity types
    entity_counts: Dict[str, int] = {}
    layers: set = set()
    for entity in msp:
        etype = entity.dxftype()
        entity_counts[etype] = entity_counts.get(etype, 0) + 1
        layer = getattr(entity.dxf, "layer", None)
        if layer:
            layers.add(str(layer))

    dimension = _detect_cad_dimension(doc)
    total = sum(entity_counts.values())

    # DXF version → human name
    version_map = {
        "AC1009": "R12", "AC1012": "R13", "AC1014": "R14",
        "AC1015": "2000", "AC1018": "2004", "AC1021": "2007",
        "AC1024": "2010", "AC1027": "2013", "AC1032": "2018",
    }
    raw_ver = getattr(doc, "dxfversion", "")
    cad_version = version_map.get(raw_ver, raw_ver)

    return {
        "pipeline":       "cad",
        "format":         "DXF",
        "cad_version":    cad_version,
        "dimension":      dimension,
        "total_entities": total,
        "entity_counts":  entity_counts,
        "layers":         sorted(layers),
        "layer_count":    len(layers),
        "capability":     "limited",
        "ux_hint":        "⚠️ Limited analysis — upload IFC for deeper insights",
    }


def _parse_dwg(file_bytes: bytes) -> dict:
    """
    DWG: ezdxf can read some DWG files in r2018 mode.
    Falls back to a minimal header scan on failure.
    """
    try:
        import ezdxf
        tmp = _save_temp(file_bytes, ".dwg")
        try:
            doc = ezdxf.readfile(tmp)
            os.unlink(tmp)
            # If it opened, treat like DXF
            result = _parse_dxf(ezdxf.write_dxf(doc).encode() if hasattr(ezdxf, "write_dxf") else b"")
            result["format"] = "DWG (converted)"
            return result
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except ImportError:
        pass

    # Fallback: detect version from DWG header magic bytes
    header = file_bytes[:6].decode("ascii", errors="replace")
    dwg_versions = {
        "AC1009": "R12", "AC1012": "R13", "AC1014": "R14",
        "AC1015": "2000", "AC1018": "2004", "AC1021": "2007",
        "AC1024": "2010", "AC1027": "2013", "AC1032": "2018",
    }
    ver = dwg_versions.get(header, f"unknown ({header})")

    return {
        "pipeline":       "cad",
        "format":         "DWG",
        "cad_version":    ver,
        "dimension":      "unknown",
        "total_entities": None,
        "entity_counts":  {},
        "layers":         [],
        "layer_count":    None,
        "note":           "DWG binary — full parse requires ODA or LibreDWG. Version detected from header only.",
        "capability":     "minimal",
        "ux_hint":        "⚠️ DWG has minimal support — convert to DXF or IFC for full analysis",
    }


def _parse_step(file_bytes: bytes) -> dict:
    """
    STEP/STP: lightweight text scan — no OCC dependency.
    Counts entity types from the DATA section.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""

    # STEP file = lines like: #123 = ENTITY_NAME(...)
    entity_pattern = re.compile(r"#\d+\s*=\s*([A-Z_]+)\s*\(")
    counts: Dict[str, int] = {}
    for match in entity_pattern.finditer(text):
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())

    # Top entities
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "pipeline":       "cad",
        "format":         "STEP",
        "dimension":      "3D",
        "total_entities": total,
        "entity_counts":  dict(top),
        "layers":         [],
        "layer_count":    0,
        "capability":     "limited",
        "ux_hint":        "⚠️ Limited analysis — upload IFC for deeper insights",
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED PARSE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_file(filename: str, file_bytes: bytes) -> dict:
    """Route to correct parser based on extension."""
    pipeline, ext = _detect_type(filename)

    if pipeline == "unsupported":
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(ALL_EXTS))}"
        )

    parse_errors: list = []

    if pipeline == "ifc":
        try:
            result = _parse_ifc(file_bytes, ext)
        except Exception as e:
            logger.error(f"[cad_ifc] IFC parse failed: {e}")
            parse_errors.append(str(e))
            result = {
                "pipeline": "ifc", "error": str(e),
                "capability": "error",
                "ux_hint": "❌ IFC parse failed — see error details",
            }
    else:
        try:
            if ext == ".dxf":
                result = _parse_dxf(file_bytes)
            elif ext == ".dwg":
                result = _parse_dwg(file_bytes)
            elif ext in (".step", ".stp"):
                result = _parse_step(file_bytes)
            else:
                raise ValueError(f"Unhandled CAD extension: {ext}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[cad_ifc] CAD parse failed: {e}")
            parse_errors.append(str(e))
            result = {
                "pipeline": "cad", "error": str(e),
                "capability": "error",
                "ux_hint": "❌ CAD parse failed — see error details",
            }

    result["filename"]   = filename
    result["file_type"]  = ext
    result["cached_at"]  = datetime.utcnow().isoformat()
    result["parse_errors"] = parse_errors
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER  (for AI reasoning)
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(summary: dict) -> str:
    """Convert parsed summary into a rich LLM context string."""
    pipeline = summary.get("pipeline", "unknown")
    context_pipeline = summary.get("context_pipeline", pipeline)
    filename = summary.get("filename", "file")
    lines = [f"═══ FILE ANALYSIS: {filename} ═══", f"Pipeline: {pipeline.upper()}"]
    if context_pipeline == "ifc" and pipeline == "cad":
        lines.append("(Context derived from auto-converted IFC)")

    if context_pipeline == "ifc":
        lines += [
            f"IFC Schema: {summary.get('schema', 'unknown')}",
            f"Total elements: {summary.get('total_elements', 0)}",
        ]
        counts = summary.get("element_counts", {})
        if counts:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
            lines.append("Element breakdown:")
            for lbl, cnt in top:
                lines.append(f"  • {lbl}: {cnt}")

        storeys = summary.get("storeys", [])
        if storeys:
            snames = [s.get("name", "?") for s in storeys]
            lines.append(f"Storeys ({len(storeys)}): {', '.join(snames)}")

        mats = summary.get("material_inventory", {})
        if mats:
            top_mats = sorted(mats.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("Materials used:")
            for mat, cnt in top_mats:
                lines.append(f"  • {mat}: {cnt} elements")

        unknown = summary.get("unknown_classes", [])
        if unknown:
            lines.append(f"⚠ Unknown IFC classes (not mapped): {', '.join(unknown[:10])}")

        missing = summary.get("missing_props_count", 0)
        if missing:
            lines.append(f"⚠ {missing} elements missing material or level data")

        # Sample elements
        samples = summary.get("elements_sample", {})
        if samples:
            lines.append("\nElement samples (first 3 per type):")
            for lbl, items in list(samples.items())[:6]:
                for el in items[:3]:
                    parts = [f"[{lbl}]"]
                    if el.get("name"):     parts.append(f"name={el['name']}")
                    if el.get("material"): parts.append(f"material={el['material']}")
                    if el.get("level"):    parts.append(f"level={el['level']}")
                    lines.append("  " + " | ".join(parts))

    elif context_pipeline == "cad":
        fmt = summary.get("format", "CAD")
        dim = summary.get("dimension", "unknown")
        lines += [
            f"Format: {fmt}",
            f"Dimension: {dim}",
            f"CAD version: {summary.get('cad_version', 'unknown')}",
            f"Total entities: {summary.get('total_entities', 'unknown')}",
        ]
        counts = summary.get("entity_counts", {})
        if counts:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
            lines.append("Entity breakdown:")
            for ename, cnt in top:
                lines.append(f"  • {ename}: {cnt}")
        layers = summary.get("layers", [])
        if layers:
            lines.append(f"Layers ({len(layers)}): {', '.join(layers[:20])}")
        note = summary.get("note")
        if note:
            lines.append(f"Note: {note}")

    err = summary.get("error")
    if err:
        lines.append(f"\n❌ Parse error: {err}")

    lines.append(f"\n{summary.get('ux_hint', '')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────────────────────────────────────────

def _judge_answer(question: str, answer: str, context: str) -> Tuple[float, str]:
    """
    Score the LLM answer on three axes:
      - Relevance  (does it address the question?)
      - Grounding  (is it backed by the file data?)
      - Completeness (key numbers / entities present?)

    Returns (score 0.0–1.0, verdict string).
    Called after every AI reasoning step; if score < 0.55 the engine retries once.
    """
    try:
        from llm_client import call_llm

        system = (
            "You are a strict QA judge for a BIM / CAD analysis assistant. "
            "Evaluate the answer below on three criteria and reply with ONLY a JSON object. "
            "No markdown fences, no explanation. "
            'Format: {"relevance": 0.0-1.0, "grounding": 0.0-1.0, "completeness": 0.0-1.0, "comment": "one sentence"} '
            "relevance: does the answer directly address the question? "
            "grounding: are claims backed by the file data provided in the context? "
            "completeness: are key numeric values / element types mentioned where expected?"
        )

        prompt = (
            f"=== FILE CONTEXT ===\n{context[:1500]}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== ANSWER ===\n{answer}\n\n"
            "Judge:"
        )

        raw = call_llm(prompt=prompt, system_prompt=system, max_tokens=150, temperature=0.0, task="classify")

        # Parse JSON
        import ast
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = None
        try:
            parsed = json.loads(clean)
        except Exception:
            m = re.search(r'\{[\s\S]*?\}', clean)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    try:
                        parsed = ast.literal_eval(m.group(0))
                    except Exception:
                        pass

        if parsed and isinstance(parsed, dict):
            r = float(parsed.get("relevance", 0.5))
            g = float(parsed.get("grounding", 0.5))
            c = float(parsed.get("completeness", 0.5))
            score = round((r * 0.4) + (g * 0.4) + (c * 0.2), 3)
            comment = parsed.get("comment", "")
            return score, comment

    except Exception as e:
        logger.warning(f"[judge] failed: {e}")

    return 0.7, "judge unavailable — default score"


# ─────────────────────────────────────────────────────────────────────────────
# AI REASONING LAYER
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are Bimlo, the AI BIM & CAD analyst of BIMLO TECHNOLOGIE — specialists in BIM engineering (3D–7D), Scan to BIM, 4D construction planning, telecom infrastructure studies, and DeepTwin AI digital twins.
You are analysing a structural or CAD file. You have access to the parsed file data below.
Answer questions factually using the data provided. Cite element counts, materials, levels, layers exactly as given. If data is missing or limited, say so clearly. Be concise and expert. Today: {today}.

FORMATTING RULES — follow these strictly:
- Always start with a **one-sentence summary** of what the file contains or what you found.
- Then use bullet points (`- `) to list specific breakdown details: element types with counts, storeys, materials, warnings.
- Wrap every filename in backticks, e.g. `TallBuilding.ifc`.
- Wrap every IFC element type or technical name in backticks, e.g. `wall`, `slab`, `Basic Wall:Outside wall`.
- Wrap every storey/level name in backticks, e.g. `Level 1`, `Level 4`.
- Each bullet = one fact. Keep bullets short and precise.
- End with a `⚠️` bullet for data quality issues (missing materials, unknown classes, parse errors) if any exist.
- Use **bold** for key numbers (e.g. **92 elements**, **5 storeys**).
- If the question is simple (yes/no, single fact), answer in 1–2 sentences only — no bullets needed.
- Do NOT write walls of prose. Structure is required.
"""

def _call_llm_with_judge(
    system_prompt: str,
    history: List[dict],
    user_msg: str,
    context: str,
    question: str,
) -> Tuple[str, float, str]:
    """
    Call LLM, judge the answer, retry once if score < 0.55.
    Returns (answer, judge_score, judge_comment).
    """
    from llm_client import call_llm, check_llm_available

    available, provider = check_llm_available()
    if not available:
        return "⚠️ LLM not configured — please set GROQ_API_KEY or CF_API_KEY.", 0.0, "no llm"

    def _call(extra_instruction: str = "") -> str:
        transcript = ""
        if history:
            for turn in history[-6:]:
                role = "User" if turn["role"] == "user" else "Bimlo"
                transcript += f"{role}: {turn['content']}\n"
            transcript += "\n"

        instruction = f"\n\nIMPORTANT: {extra_instruction}" if extra_instruction else ""
        prompt = f"{transcript}User: {user_msg}{instruction}\nBimlo:"
        return call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=900,
            temperature=0.3,
        ).strip()

    answer = _call()
    score, comment = _judge_answer(question, answer, context)
    logger.info(f"[judge] score={score} comment={comment!r}")

    if score < 0.55:
        logger.info("[judge] score below threshold — retrying with stricter instruction")
        answer = _call(
            "Your previous answer was incomplete or not well grounded in the file data. "
            "Use specific numbers, element types, and layer/material names from the context. "
            "Be precise."
        )
        score, comment = _judge_answer(question, answer, context)
        logger.info(f"[judge] retry score={score} comment={comment!r}")

    return answer, score, comment


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CadQueryRequest(BaseModel):
    query:      str
    file_id:    str
    session_id: Optional[str] = None


class CadQueryResponse(BaseModel):
    answer:        str
    session_id:    str
    file_id:       str
    judge_score:   float
    judge_comment: str
    pipeline:      str
    ux_hint:       str


class CadUploadResponse(BaseModel):
    file_id:        str
    filename:       str
    pipeline:       str
    ux_hint:        str
    schema:         Optional[str]   = None
    total_elements: Optional[int]   = None
    total_entities: Optional[int]   = None
    dimension:      Optional[str]   = None
    storeys:        Optional[List]  = None
    element_counts: Optional[dict]  = None
    entity_counts:  Optional[dict]  = None
    layers:         Optional[List]  = None
    material_inventory: Optional[dict] = None
    parse_errors:   List[str]       = []
    cached_at:      str             = ""
    converted_entities: Optional[int] = None
    ifc_available:  bool            = False   # True if silent CAD→IFC conversion succeeded


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/cad/upload", response_model=CadUploadResponse)
async def cad_upload(file: UploadFile = File(...)):
    """
    Upload a CAD/IFC file. Parses it immediately and caches the result.
    Returns a file_id to use in subsequent /api/cad/query calls.
    """
    filename  = file.filename or "upload"
    pipeline, ext = _detect_type(filename)

    if pipeline == "unsupported":
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(ALL_EXTS))}"
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    logger.info(f"[cad_upload] {filename} | {len(file_bytes)} bytes | pipeline={pipeline}")

    summary = _parse_file(filename, file_bytes)
    file_id = str(uuid.uuid4())
    summary["file_id"] = file_id

    # ── Silent CAD → IFC conversion (internal, not exposed to user) ──────────
    ifc_available = False
    converted_entities: Optional[int] = None
    if pipeline == "cad" and _CAD_TO_IFC_AVAILABLE:
        try:
            ifc_bytes, conv_report = convert_cad_to_ifc(filename, file_bytes)
            summary["_ifc_conversion_report"] = conv_report
            converted_entities = int(conv_report.get("cleaned_entities") or 0)
            if converted_entities > 0:
                summary["_ifc_bytes"] = ifc_bytes
                ifc_available = True
                # Re-parse the converted IFC to get rich context (elements, storeys, materials)
                try:
                    ifc_summary = _parse_ifc(ifc_bytes, ".ifc")
                    # Merge IFC context into summary, preserving CAD identity fields
                    for key in ("element_counts", "total_elements", "storeys",
                                "material_inventory", "elements_sample", "embed_strings",
                                "unknown_classes", "missing_props_count", "schema"):
                        if key in ifc_summary:
                            summary[key] = ifc_summary[key]
                    # Keep pipeline as "cad" so upload response is correct, but tag context source
                    summary["context_pipeline"] = "ifc"
                    logger.info(f"[cad_upload] IFC context merged | elements={ifc_summary.get('total_elements', 0)}")
                except Exception as ifc_parse_err:
                    logger.warning(f"[cad_upload] IFC context merge failed: {ifc_parse_err}")
                if summary.get("error") or summary.get("parse_errors"):
                    summary["ux_hint"] = (
                        "Preview ready from converted IFC. Native CAD parsing failed for this file."
                    )
                logger.info(
                    f"[cad_upload] silent IFC conversion OK | "
                    f"entities={converted_entities} | "
                    f"ifc_size={conv_report.get('ifc_size_bytes')} bytes"
                )
            else:
                summary.pop("_ifc_bytes", None)
                summary.setdefault("parse_errors", []).append(
                    "No previewable geometry could be extracted for CAD to IFC conversion."
                )
                if Path(filename).suffix.lower() == ".dwg":
                    summary["ux_hint"] = (
                        "DWG uploaded, but no geometry could be extracted. "
                        "Current DWG support is minimal; convert to DXF or IFC for preview."
                    )
                logger.warning(
                    f"[cad_upload] silent IFC conversion produced no geometry | "
                    f"entities={converted_entities} | "
                    f"ifc_size={conv_report.get('ifc_size_bytes')} bytes"
                )
        except Exception as conv_err:
            logger.warning(f"[cad_upload] silent IFC conversion failed: {conv_err}")

    CadSharedContext.cache_file(file_id, summary)

    logger.info(
        f"[cad_upload] done | file_id={file_id} | pipeline={pipeline} "
        f"| ux={summary.get('ux_hint', '')} | ifc_available={ifc_available}"
    )

    return CadUploadResponse(
        file_id        = file_id,
        filename       = filename,
        pipeline       = summary.get("pipeline", pipeline),
        ux_hint        = summary.get("ux_hint", ""),
        schema         = summary.get("schema"),
        total_elements = summary.get("total_elements"),
        total_entities = summary.get("total_entities"),
        dimension      = summary.get("dimension"),
        storeys        = summary.get("storeys"),
        element_counts = summary.get("element_counts"),
        entity_counts  = summary.get("entity_counts"),
        layers         = summary.get("layers"),
        material_inventory = summary.get("material_inventory"),
        parse_errors   = summary.get("parse_errors", []),
        cached_at      = summary.get("cached_at", ""),
        ifc_available  = ifc_available,
        converted_entities = converted_entities,
    )


@router.post("/api/cad/query")
async def cad_query(req: CadQueryRequest):
    """
    Ask a question about a previously uploaded CAD/IFC file.
    Streams Server-Sent Events with real status updates as each step executes:
      { "type": "status", "node": "...", "icon": "...", "message": "..." }
      { "type": "result", "answer": "...", "session_id": "...", ... }
      { "type": "error",  "message": "..." }
    """
    summary = CadSharedContext.get_file(req.file_id)
    if not summary:
        raise HTTPException(404, f"file_id '{req.file_id}' not found — upload the file first.")

    sid     = req.session_id or str(uuid.uuid4())
    history = CadSharedContext.get_history(sid)
    CadSharedContext.set_summary(sid, summary)

    q: queue.Queue = queue.Queue()
    DONE = object()

    def emit(node: str, icon: str, message: str):
        q.put({"type": "status", "node": node, "icon": icon, "message": message})

    def run():
        try:
            # ── Step 1: file lookup ────────────────────────────────────────
            fname    = summary.get("filename", "file")
            pipeline = summary.get("pipeline", "unknown").upper()
            n_elem   = summary.get("total_elements") or summary.get("total_entities") or 0
            emit("file_lookup", "📂", f"Loaded `{fname}` — {n_elem} elements via {pipeline} pipeline")

            # ── Step 2: context build ──────────────────────────────────────
            emit("context_build", "🔍", "Extracting element structure, materials & storeys")
            context = _build_context_block(summary)

            system_prompt = _SYSTEM_PROMPT.format(
                today=datetime.utcnow().strftime("%B %d, %Y")
            ) + f"\n\n{context}"

            # ── Step 3: element summary ────────────────────────────────────
            counts  = summary.get("element_counts") or summary.get("entity_counts") or {}
            top3    = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in top3) if top3 else "no typed elements"
            emit("element_scan", "🏗️", f"Top elements: {top_str}")

            # ── Step 4: storey / layer context ─────────────────────────────
            storeys = summary.get("storeys", [])
            layers  = summary.get("layers", [])
            if storeys:
                snames = [s.get("name", "?") for s in storeys]
                emit("spatial_context", "📐", f"Spatial context: {len(storeys)} storeys — {', '.join(snames[:4])}")
            elif layers:
                emit("spatial_context", "📐", f"Layer context: {len(layers)} CAD layers detected")

            # ── Step 5: material scan ──────────────────────────────────────
            mats = summary.get("material_inventory", {})
            if mats:
                top_mat = sorted(mats.items(), key=lambda x: x[1], reverse=True)[0]
                emit("material_scan", "🧱", f"Materials found — most used: `{top_mat[0]}` ({top_mat[1]} elements)")
            else:
                emit("material_scan", "🧱", "Material data sparse or missing in this file")

            # ── Step 6: LLM reasoning ──────────────────────────────────────
            emit("llm_reason", "🧠", f"Reasoning over {pipeline} data to answer your question")

            answer, judge_score, judge_comment = _call_llm_with_judge(
                system_prompt=system_prompt,
                history=history,
                user_msg=req.query,
                context=context,
                question=req.query,
            )

            # ── Step 7: judge verdict ──────────────────────────────────────
            score_pct = round(judge_score * 100)
            verdict   = "✅" if judge_score >= 0.7 else "⚠️"
            emit("llm_judge", verdict, f"Answer quality: {score_pct}%{' — ' + judge_comment if judge_comment else ''}")

            CadSharedContext.append_turn(sid, "user",      req.query)
            CadSharedContext.append_turn(sid, "assistant", answer)

            # ── Step 8: sync to main session memory ───────────────────────
            # Push this turn + IFC/CAD summary into main.py _sessions and
            # SharedContext so the RAG engine, report agent, and intent
            # classifier all know what the CAD agent just said.
            if _BRIDGE_AVAILABLE:
                _sync_to_main(
                    session_id       = sid,
                    user_query       = req.query,
                    assistant_answer = answer,
                    file_summary     = {**summary, "file_id": req.file_id},
                )

            q.put({
                "type":          "result",
                "answer":        answer,
                "session_id":    sid,
                "file_id":       req.file_id,
                "judge_score":   judge_score,
                "judge_comment": judge_comment,
                "pipeline":      summary.get("pipeline", "unknown"),
                "ux_hint":       summary.get("ux_hint", ""),
            })
        except Exception as e:
            logger.error(f"[cad_query] error: {e}")
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(DONE)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=300))
            except Exception:
                break
            if item is DONE:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/api/cad/files")
async def cad_list_files():
    """List all cached CAD/IFC file summaries."""
    return {"files": CadSharedContext.list_files(), "total": len(CadSharedContext.list_files())}


@router.delete("/api/cad/files/{file_id}")
async def cad_delete_file(file_id: str):
    """Remove a file from the cache."""
    if not CadSharedContext.get_file(file_id):
        raise HTTPException(404, f"file_id '{file_id}' not found.")
    CadSharedContext.delete_file(file_id)
    return {"status": "deleted", "file_id": file_id}


@router.get("/api/cad/files/{file_id}/ifc")
async def cad_download_ifc(file_id: str):
    """
    Download the silently-converted IFC file for a previously uploaded CAD file.
    Only available when the source was a CAD file (.dxf / .dwg / .step).
    Note: CAD files may contain some inaccuracies in the converted IFC.
    """
    summary = CadSharedContext.get_file(file_id)
    if not summary:
        raise HTTPException(404, f"file_id '{file_id}' not found.")

    ifc_bytes = summary.get("_ifc_bytes")
    if not ifc_bytes:
        raise HTTPException(
            404,
            "No IFC conversion available for this file. "
            "This may be because the source was already an IFC, "
            "the conversion failed, or ifcopenshell is not installed."
        )

    original_name = Path(summary.get("filename", "export")).stem
    download_name = f"{original_name}_converted.ifc"

    return Response(
        content=ifc_bytes,
        media_type="application/x-step",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )