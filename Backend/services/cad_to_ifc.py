"""
cad_to_ifc.py
─────────────────────────────────────────────────────────────────────────────
Internal CAD → IFC conversion pipeline for BIMLO Copilot.
Transparent to the user — runs silently on every CAD upload.

Pipeline
────────
  0  Detect & classify input  (.dxf / .dwg / .step)
  1  Extract raw geometry     (ezdxf or text scan)
  2  Normalize geometry       → unified entity list
  3  Clean geometry           (merge collinear, dedupe, close polygons)
  4  Infer semantics          (layer-based → geometry heuristics → AI fallback)
  5  Build BIM structure      (walls / spaces / columns / openings)
  6  Generate IFC             (IfcOpenShell)
  7  Attach geometry          (extruded solids, 2D→3D extrusion)
  8  Add default properties   (material, height, etc.)
  9  Export .ifc bytes
"""

from __future__ import annotations

import io
import os
import re
import math
import logging
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cad_to_ifc")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WALL_HEIGHT   = 3.0   # metres
DEFAULT_WALL_MATERIAL = "Concrete"
DEFAULT_SLAB_HEIGHT   = 0.3
DEFAULT_COL_RADIUS    = 0.2
EXTRUDE_HEIGHT        = 3.0   # fallback extrusion for 2D→3D

# Layer-name keyword → semantic type
LAYER_SEMANTICS: Dict[str, str] = {
    "wall":    "wall",
    "mur":     "wall",      # French
    "jdar":    "wall",      # Arabic transliteration
    "cloison": "wall",
    "partition": "wall",
    "slab":    "slab",
    "dalle":   "slab",
    "floor":   "slab",
    "plancher":"slab",
    "column":  "column",
    "col":     "column",
    "pilier":  "column",
    "pillar":  "column",
    "door":    "door",
    "porte":   "door",
    "window":  "window",
    "fenetre": "window",
    "vitre":   "window",
    "beam":    "beam",
    "poutre":  "beam",
    "stair":   "stair",
    "escalier":"stair",
    "room":    "space",
    "space":   "space",
    "piece":   "space",
    "axis":    None,        # skip axis / grid layers
    "grid":    None,
    "grille":  None,
    "dim":     None,        # dimensions / annotations
    "text":    None,
    "hatch":   None,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_input_type(filename: str) -> str:
    """Returns one of: 'dxf', 'dwg', 'step', 'unknown'"""
    ext = Path(filename).suffix.lower()
    return {".dxf": "dxf", ".dwg": "dwg", ".step": "step", ".stp": "step"}.get(ext, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RAW GEOMETRY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dxf(file_bytes: bytes) -> List[dict]:
    """Extract raw entities from DXF file bytes."""
    try:
        import ezdxf
    except ImportError:
        raise RuntimeError("ezdxf not installed")

    fd, tmp = tempfile.mkstemp(suffix=".dxf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        doc = ezdxf.readfile(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    entities = []
    msp = doc.modelspace()

    for e in msp:
        etype = e.dxftype()
        layer = getattr(e.dxf, "layer", "0")

        if etype == "LINE":
            try:
                s = e.dxf.start
                en = e.dxf.end
                entities.append({
                    "type":  "line",
                    "layer": layer,
                    "start": [round(s.x, 6), round(s.y, 6)],
                    "end":   [round(en.x, 6), round(en.y, 6)],
                })
            except Exception:
                pass

        elif etype in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts = []
                if etype == "LWPOLYLINE":
                    for p in e.get_points():
                        pts.append([round(p[0], 6), round(p[1], 6)])
                else:
                    for v in e.vertices:
                        pts.append([round(v.dxf.location.x, 6), round(v.dxf.location.y, 6)])
                if len(pts) >= 2:
                    entities.append({
                        "type":   "polyline",
                        "layer":  layer,
                        "points": pts,
                        "closed": bool(getattr(e.dxf, "flags", 0) & 1) or getattr(e, "closed", False),
                    })
            except Exception:
                pass

        elif etype == "CIRCLE":
            try:
                c = e.dxf.center
                entities.append({
                    "type":   "circle",
                    "layer":  layer,
                    "center": [round(c.x, 6), round(c.y, 6)],
                    "radius": round(e.dxf.radius, 6),
                })
            except Exception:
                pass

        elif etype == "ARC":
            try:
                c = e.dxf.center
                entities.append({
                    "type":        "arc",
                    "layer":       layer,
                    "center":      [round(c.x, 6), round(c.y, 6)],
                    "radius":      round(e.dxf.radius, 6),
                    "start_angle": round(e.dxf.start_angle, 4),
                    "end_angle":   round(e.dxf.end_angle, 4),
                })
            except Exception:
                pass

        elif etype in ("INSERT",):
            # Block inserts — record as reference point
            try:
                ins = e.dxf.insert
                entities.append({
                    "type":       "insert",
                    "layer":      layer,
                    "block_name": getattr(e.dxf, "name", ""),
                    "position":   [round(ins.x, 6), round(ins.y, 6)],
                })
            except Exception:
                pass

    return entities


def _extract_dwg(file_bytes: bytes) -> List[dict]:
    """
    DWG extraction: try ezdxf first (sometimes works on newer DWG).
    Falls back to empty list with a warning.
    """
    try:
        import ezdxf
        fd, tmp = tempfile.mkstemp(suffix=".dwg")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)
            doc = ezdxf.readfile(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        # Write to DXF in-memory and re-parse
        buf = io.StringIO()
        doc.write(buf)
        dxf_bytes = buf.getvalue().encode("utf-8")
        return _extract_dxf(dxf_bytes)
    except Exception as ex:
        logger.warning(f"[cad_to_ifc] DWG extraction partial/failed: {ex}")
        return []


def _extract_step(file_bytes: bytes) -> List[dict]:
    """
    STEP: limited — extract entity names only (no OCC dependency).
    We create placeholder 'step_entity' entries for BIM mapping.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return []

    pattern = re.compile(r"#(\d+)\s*=\s*([A-Z_]+)\s*\(")
    entities = []
    for m in pattern.finditer(text):
        entities.append({
            "type":   "step_entity",
            "layer":  "default",
            "entity_type": m.group(2),
            "step_id": int(m.group(1)),
        })
    return entities


def extract_geometry(filename: str, file_bytes: bytes) -> List[dict]:
    """Route to correct extractor."""
    kind = detect_input_type(filename)
    if kind == "dxf":
        return _extract_dxf(file_bytes)
    elif kind == "dwg":
        return _extract_dwg(file_bytes)
    elif kind == "step":
        return _extract_step(file_bytes)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE TO UNIFIED FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entities(raw: List[dict]) -> List[dict]:
    """
    Convert all raw entities to the unified internal format:
    {type, layer, start, end}  for line-like
    {type, layer, points}      for polygon-like
    {type, layer, center, radius} for circular
    """
    normalized = []
    for e in raw:
        t = e.get("type", "")

        if t == "line":
            normalized.append({
                "type":  "line",
                "layer": e.get("layer", "0"),
                "start": e["start"],
                "end":   e["end"],
            })

        elif t == "polyline":
            pts = e.get("points", [])
            normalized.append({
                "type":   "polyline",
                "layer":  e.get("layer", "0"),
                "points": pts,
                "closed": e.get("closed", False),
            })

        elif t == "circle":
            normalized.append({
                "type":   "circle",
                "layer":  e.get("layer", "0"),
                "center": e["center"],
                "radius": e["radius"],
            })

        elif t == "arc":
            normalized.append({
                "type":   "arc",
                "layer":  e.get("layer", "0"),
                "center": e["center"],
                "radius": e["radius"],
            })

        elif t == "insert":
            # Treat inserts as point objects
            normalized.append({
                "type":   "point",
                "layer":  e.get("layer", "0"),
                "coords": e.get("position", [0, 0]),
            })

        elif t == "step_entity":
            normalized.append(e)

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GEOMETRY CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _pt_dist(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _are_collinear(l1: dict, l2: dict, tol: float = 1e-4) -> bool:
    """Check if two lines are collinear and share/touch an endpoint."""
    # Must share an endpoint within tol
    shared = False
    for ep1 in [l1["start"], l1["end"]]:
        for ep2 in [l2["start"], l2["end"]]:
            if _pt_dist(ep1, ep2) < tol:
                shared = True
                break

    if not shared:
        return False

    # Check direction vectors parallel
    def vec(l):
        return [l["end"][0]-l["start"][0], l["end"][1]-l["start"][1]]

    v1, v2 = vec(l1), vec(l2)
    cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
    mag = (math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)) + 1e-12
    return (cross / mag) < tol


def _merge_collinear(lines: List[dict]) -> List[dict]:
    """Merge collinear connected line segments."""
    merged = list(lines)
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        result = []
        for i in range(len(merged)):
            if used[i]:
                continue
            l1 = merged[i]
            for j in range(i+1, len(merged)):
                if used[j]:
                    continue
                l2 = merged[j]
                if l1["layer"] == l2["layer"] and _are_collinear(l1, l2):
                    # Merge: find the two furthest endpoints
                    pts = [l1["start"], l1["end"], l2["start"], l2["end"]]
                    max_d = 0
                    best = (pts[0], pts[1])
                    for a in pts:
                        for b in pts:
                            d = _pt_dist(a, b)
                            if d > max_d:
                                max_d = d
                                best = (a, b)
                    l1 = {"type": "line", "layer": l1["layer"], "start": best[0], "end": best[1]}
                    used[j] = True
                    changed = True
            result.append(l1)
            used[i] = True
        merged = result
    return merged


def _dedup_lines(lines: List[dict], tol: float = 1e-4) -> List[dict]:
    """Remove duplicate or near-zero-length lines."""
    seen = []
    for l in lines:
        if _pt_dist(l["start"], l["end"]) < tol:
            continue   # zero-length
        duplicate = False
        for s in seen:
            if ((_pt_dist(l["start"], s["start"]) < tol and _pt_dist(l["end"], s["end"]) < tol) or
                (_pt_dist(l["start"], s["end"]) < tol and _pt_dist(l["end"], s["start"]) < tol)):
                duplicate = True
                break
        if not duplicate:
            seen.append(l)
    return seen


def clean_geometry(entities: List[dict]) -> List[dict]:
    """
    Run all geometry cleaning steps:
    - Deduplicate lines
    - Merge collinear segments
    - Keep non-line entities as-is
    """
    lines   = [e for e in entities if e["type"] == "line"]
    others  = [e for e in entities if e["type"] != "line"]

    lines = _dedup_lines(lines)
    lines = _merge_collinear(lines)

    return lines + others


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SEMANTIC INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _infer_from_layer(layer: str) -> Optional[str]:
    """Method A — layer-name keyword match."""
    low = layer.lower()
    for keyword, sem_type in LAYER_SEMANTICS.items():
        if keyword in low:
            return sem_type   # None = skip
    return "unknown"


def _line_length(e: dict) -> float:
    if e["type"] == "line":
        return _pt_dist(e["start"], e["end"])
    return 0.0


def _infer_from_geometry(entity: dict, all_entities: List[dict]) -> str:
    """Method B — geometry heuristics."""
    t = entity["type"]

    if t == "circle":
        return "column"   # circles → columns

    if t == "line":
        length = _line_length(entity)
        if length > 0.5:   # longer lines → walls (unit-agnostic heuristic)
            return "wall"
        return "unknown"

    if t == "polyline":
        pts = entity.get("points", [])
        if entity.get("closed") and len(pts) >= 3:
            # Compute area (shoelace)
            area = 0.0
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            area = abs(area) / 2.0
            if area > 1.0:
                return "space"   # large closed polygon → room/space
            return "wall"        # small closed polygon → wall outline
        return "wall"

    if t == "step_entity":
        et = entity.get("entity_type", "")
        if "WALL" in et or "SLAB" in et:
            return "wall"
        if "COLUMN" in et or "PILLAR" in et:
            return "column"
        return "unknown"

    return "unknown"


def infer_semantics(entities: List[dict]) -> List[dict]:
    """
    Attach 'semantic_type' to each entity.
    Priority: layer → geometry → 'unknown'
    Entities with None semantic (axis, dim, text) are filtered out.
    """
    result = []
    for e in entities:
        layer = e.get("layer", "0")
        sem = _infer_from_layer(layer)

        if sem is None:
            continue   # Skip annotation/axis layers

        if sem == "unknown":
            sem = _infer_from_geometry(e, entities)

        e = dict(e)
        e["semantic_type"] = sem
        result.append(e)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD BIM STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_bim_structure(entities: List[dict]) -> Dict[str, List[dict]]:
    """
    Group semantic entities into BIM categories.
    Returns: {walls: [...], spaces: [...], columns: [...], doors: [...], windows: [...], ...}
    """
    bim: Dict[str, List[dict]] = defaultdict(list)

    for e in entities:
        sem = e.get("semantic_type", "unknown")
        if sem == "unknown":
            sem = "misc"
        bim[sem + "s" if not sem.endswith("s") else sem].append(e)

    # Add relationships: assign each entity to level "Level 1" (single-level MVP)
    for category in bim.values():
        for e in category:
            e.setdefault("level", "Level 1")

    return dict(bim)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6–9 — IFC GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _make_ifc_boilerplate(model) -> tuple:
    """Create standard IFC project hierarchy. Returns (project, site, building, storey, owner_history)."""
    import ifcopenshell
    import ifcopenshell.api

    # Owner history (required by spec)
    person       = model.create_entity("IfcPerson", FamilyName="BIMLO")
    organisation = model.create_entity("IfcOrganization", Name="BIMLO TECHNOLOGIE")
    person_org   = model.create_entity("IfcPersonAndOrganization",
                                       ThePerson=person, TheOrganization=organisation)
    application  = model.create_entity("IfcApplication",
                                       ApplicationDeveloper=organisation,
                                       Version="1.0",
                                       ApplicationFullName="BIMLO CAD Converter",
                                       ApplicationIdentifier="BIMLO_CAD")
    timestamp    = 0  # epoch
    owner_history = model.create_entity("IfcOwnerHistory",
                                        OwningUser=person_org,
                                        OwningApplication=application,
                                        ChangeAction="ADDED",
                                        CreationDate=timestamp)

    # Units
    length_unit = model.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    unit_assign = model.create_entity("IfcUnitAssignment", Units=[length_unit])

    # Geometric representation context
    geom_ctx = model.create_entity("IfcGeometricRepresentationContext",
                                   ContextIdentifier="Body",
                                   ContextType="Model",
                                   CoordinateSpaceDimension=3,
                                   Precision=1e-5,
                                   WorldCoordinateSystem=model.create_entity(
                                       "IfcAxis2Placement3D",
                                       Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
                                   ))

    # Project
    project = model.create_entity("IfcProject",
                                  GlobalId=ifcopenshell.guid.new(),
                                  OwnerHistory=owner_history,
                                  Name="CAD Import",
                                  UnitsInContext=unit_assign,
                                  RepresentationContexts=[geom_ctx])

    # Site
    site_placement = model.create_entity("IfcLocalPlacement",
                                         RelativePlacement=model.create_entity(
                                             "IfcAxis2Placement3D",
                                             Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0,0.0,0.0))
                                         ))
    site = model.create_entity("IfcSite",
                                GlobalId=ifcopenshell.guid.new(),
                                OwnerHistory=owner_history,
                                Name="Site",
                                ObjectPlacement=site_placement)

    # Building
    building_placement = model.create_entity("IfcLocalPlacement",
                                              PlacementRelTo=site_placement,
                                              RelativePlacement=model.create_entity(
                                                  "IfcAxis2Placement3D",
                                                  Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0,0.0,0.0))
                                              ))
    building = model.create_entity("IfcBuilding",
                                   GlobalId=ifcopenshell.guid.new(),
                                   OwnerHistory=owner_history,
                                   Name="Building",
                                   ObjectPlacement=building_placement)

    # Storey
    storey_placement = model.create_entity("IfcLocalPlacement",
                                            PlacementRelTo=building_placement,
                                            RelativePlacement=model.create_entity(
                                                "IfcAxis2Placement3D",
                                                Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0,0.0,0.0))
                                            ))
    storey = model.create_entity("IfcBuildingStorey",
                                 GlobalId=ifcopenshell.guid.new(),
                                 OwnerHistory=owner_history,
                                 Name="Level 1",
                                 ObjectPlacement=storey_placement,
                                 Elevation=0.0)

    # Aggregate relationships
    model.create_entity("IfcRelAggregates",
                        GlobalId=ifcopenshell.guid.new(),
                        OwnerHistory=owner_history,
                        RelatingObject=project,
                        RelatedObjects=[site])
    model.create_entity("IfcRelAggregates",
                        GlobalId=ifcopenshell.guid.new(),
                        OwnerHistory=owner_history,
                        RelatingObject=site,
                        RelatedObjects=[building])
    model.create_entity("IfcRelAggregates",
                        GlobalId=ifcopenshell.guid.new(),
                        OwnerHistory=owner_history,
                        RelatingObject=building,
                        RelatedObjects=[storey])

    return project, site, building, storey, owner_history, geom_ctx


def _make_placement(model, x: float, y: float, z: float = 0.0, parent_placement=None):
    """Create a local placement at (x, y, z)."""
    axis2 = model.create_entity("IfcAxis2Placement3D",
                                 Location=model.create_entity("IfcCartesianPoint",
                                                              Coordinates=(x, y, z)))
    kwargs = {"RelativePlacement": axis2}
    if parent_placement:
        kwargs["PlacementRelTo"] = parent_placement
    return model.create_entity("IfcLocalPlacement", **kwargs)


def _make_extruded_solid(model, profile, height: float, geom_ctx):
    """
    Create an IfcExtrudedAreaSolid from a 2D profile.
    profile: IfcProfileDef (e.g. IfcRectangleProfileDef or IfcArbitraryClosedProfileDef)
    """
    direction = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    position  = model.create_entity("IfcAxis2Placement3D",
                                    Location=model.create_entity("IfcCartesianPoint",
                                                                 Coordinates=(0.0, 0.0, 0.0)))
    solid = model.create_entity("IfcExtrudedAreaSolid",
                                SweptArea=profile,
                                Position=position,
                                ExtrudedDirection=direction,
                                Depth=height)

    body_repr = model.create_entity("IfcShapeRepresentation",
                                    ContextOfItems=geom_ctx,
                                    RepresentationIdentifier="Body",
                                    RepresentationType="SweptSolid",
                                    Items=[solid])
    return model.create_entity("IfcProductDefinitionShape",
                               Representations=[body_repr])


def _add_wall(model, entity: dict, owner_history, storey_placement, geom_ctx) -> Any:
    """Convert a line/polyline entity → IfcWall with extruded geometry."""
    import ifcopenshell

    # Determine start/end from entity type
    if entity["type"] == "line":
        sx, sy = entity["start"]
        ex, ey = entity["end"]
    elif entity["type"] == "polyline":
        pts = entity.get("points", [[0,0],[1,0]])
        sx, sy = pts[0]
        ex, ey = pts[-1]
    else:
        sx, sy, ex, ey = 0.0, 0.0, 1.0, 0.0

    length = max(_pt_dist([sx,sy],[ex,ey]), 0.01)
    angle  = math.atan2(ey - sy, ex - sx)

    # Placement at start point, rotated along wall direction
    dir_x  = math.cos(angle)
    dir_y  = math.sin(angle)
    axis   = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    ref    = model.create_entity("IfcDirection", DirectionRatios=(dir_x, dir_y, 0.0))
    loc    = model.create_entity("IfcCartesianPoint", Coordinates=(sx, sy, 0.0))
    place3 = model.create_entity("IfcAxis2Placement3D", Location=loc, Axis=axis, RefDirection=ref)
    placement = model.create_entity("IfcLocalPlacement",
                                    PlacementRelTo=storey_placement,
                                    RelativePlacement=place3)

    # Rectangle profile: length × 0.2m (wall thickness default)
    thickness = 0.2
    profile   = model.create_entity("IfcRectangleProfileDef",
                                    ProfileType="AREA",
                                    XDim=length,
                                    YDim=thickness)
    shape = _make_extruded_solid(model, profile, DEFAULT_WALL_HEIGHT, geom_ctx)

    wall = model.create_entity("IfcWall",
                               GlobalId=ifcopenshell.guid.new(),
                               OwnerHistory=owner_history,
                               Name=f"Wall_{entity.get('layer','0')}",
                               ObjectPlacement=placement,
                               Representation=shape)
    return wall


def _add_column(model, entity: dict, owner_history, storey_placement, geom_ctx) -> Any:
    """Convert a circle entity → IfcColumn."""
    import ifcopenshell

    cx, cy = entity.get("center", [0.0, 0.0])
    radius = entity.get("radius", DEFAULT_COL_RADIUS)

    loc    = model.create_entity("IfcCartesianPoint", Coordinates=(cx, cy, 0.0))
    place3 = model.create_entity("IfcAxis2Placement3D", Location=loc)
    placement = model.create_entity("IfcLocalPlacement",
                                    PlacementRelTo=storey_placement,
                                    RelativePlacement=place3)

    profile = model.create_entity("IfcCircleProfileDef",
                                  ProfileType="AREA",
                                  Radius=radius)
    shape = _make_extruded_solid(model, profile, DEFAULT_WALL_HEIGHT, geom_ctx)

    column = model.create_entity("IfcColumn",
                                 GlobalId=ifcopenshell.guid.new(),
                                 OwnerHistory=owner_history,
                                 Name=f"Column_{entity.get('layer','0')}",
                                 ObjectPlacement=placement,
                                 Representation=shape)
    return column


def _add_slab(model, entity: dict, owner_history, storey_placement, geom_ctx) -> Any:
    """Convert a closed polyline → IfcSlab."""
    import ifcopenshell

    pts = entity.get("points", [])
    if len(pts) < 3:
        return None

    # Build IfcArbitraryClosedProfileDef
    ifc_pts = [model.create_entity("IfcCartesianPoint", Coordinates=(p[0], p[1]))
               for p in pts]
    polyline = model.create_entity("IfcPolyline", Points=ifc_pts + [ifc_pts[0]])
    profile  = model.create_entity("IfcArbitraryClosedProfileDef",
                                   ProfileType="AREA",
                                   OuterCurve=polyline)

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    loc    = model.create_entity("IfcCartesianPoint", Coordinates=(cx, cy, 0.0))
    place3 = model.create_entity("IfcAxis2Placement3D", Location=loc)
    placement = model.create_entity("IfcLocalPlacement",
                                    PlacementRelTo=storey_placement,
                                    RelativePlacement=place3)

    shape = _make_extruded_solid(model, profile, DEFAULT_SLAB_HEIGHT, geom_ctx)

    slab = model.create_entity("IfcSlab",
                               GlobalId=ifcopenshell.guid.new(),
                               OwnerHistory=owner_history,
                               Name=f"Slab_{entity.get('layer','0')}",
                               ObjectPlacement=placement,
                               Representation=shape)
    return slab


def _add_space(model, entity: dict, owner_history, storey_placement, geom_ctx) -> Any:
    """Convert a large closed polyline → IfcSpace."""
    import ifcopenshell

    pts = entity.get("points", [])
    if len(pts) < 3:
        return None

    ifc_pts = [model.create_entity("IfcCartesianPoint", Coordinates=(p[0], p[1]))
               for p in pts]
    polyline = model.create_entity("IfcPolyline", Points=ifc_pts + [ifc_pts[0]])
    profile  = model.create_entity("IfcArbitraryClosedProfileDef",
                                   ProfileType="AREA",
                                   OuterCurve=polyline)

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    loc    = model.create_entity("IfcCartesianPoint", Coordinates=(cx, cy, 0.0))
    place3 = model.create_entity("IfcAxis2Placement3D", Location=loc)
    placement = model.create_entity("IfcLocalPlacement",
                                    PlacementRelTo=storey_placement,
                                    RelativePlacement=place3)

    shape = _make_extruded_solid(model, profile, EXTRUDE_HEIGHT, geom_ctx)

    space = model.create_entity("IfcSpace",
                                GlobalId=ifcopenshell.guid.new(),
                                OwnerHistory=owner_history,
                                Name=f"Space_{entity.get('layer','0')}",
                                ObjectPlacement=placement,
                                Representation=shape)
    return space


def _add_property_set(model, element, owner_history, material: str, height: float):
    """Attach a basic Pset_Common property set with material and height."""
    import ifcopenshell

    nom_height = model.create_entity("IfcPropertySingleValue",
                                     Name="Height",
                                     NominalValue=model.create_entity("IfcLengthMeasure", wrappedValue=height))
    nom_mat    = model.create_entity("IfcPropertySingleValue",
                                     Name="Material",
                                     NominalValue=model.create_entity("IfcLabel", wrappedValue=material))
    pset = model.create_entity("IfcPropertySet",
                               GlobalId=ifcopenshell.guid.new(),
                               OwnerHistory=owner_history,
                               Name="Pset_CADImport",
                               HasProperties=[nom_height, nom_mat])
    model.create_entity("IfcRelDefinesByProperties",
                        GlobalId=ifcopenshell.guid.new(),
                        OwnerHistory=owner_history,
                        RelatedObjects=[element],
                        RelatingPropertyDefinition=pset)


def generate_ifc(bim_structure: Dict[str, List[dict]]) -> bytes:
    """
    Steps 6–9: Build the full IFC model and return as bytes.
    Silently skips entities it can't handle.
    """
    try:
        import ifcopenshell
    except ImportError:
        raise RuntimeError("ifcopenshell not installed — run: pip install ifcopenshell")

    model = ifcopenshell.file(schema="IFC4")
    project, site, building, storey, owner_history, geom_ctx = _make_ifc_boilerplate(model)
    storey_placement = storey.ObjectPlacement

    contained_elements = []

    # ── Walls ────────────────────────────────────────────────────────────────
    for e in bim_structure.get("walls", []):
        if e["type"] in ("line", "polyline"):
            try:
                wall = _add_wall(model, e, owner_history, storey_placement, geom_ctx)
                _add_property_set(model, wall, owner_history, DEFAULT_WALL_MATERIAL, DEFAULT_WALL_HEIGHT)
                contained_elements.append(wall)
            except Exception as ex:
                logger.debug(f"[ifc_gen] wall skip: {ex}")

    # ── Columns ──────────────────────────────────────────────────────────────
    for e in bim_structure.get("columns", []):
        if e["type"] == "circle":
            try:
                col = _add_column(model, e, owner_history, storey_placement, geom_ctx)
                _add_property_set(model, col, owner_history, "Concrete", DEFAULT_WALL_HEIGHT)
                contained_elements.append(col)
            except Exception as ex:
                logger.debug(f"[ifc_gen] column skip: {ex}")

    # ── Slabs ────────────────────────────────────────────────────────────────
    for e in bim_structure.get("slabs", []):
        if e["type"] == "polyline" and e.get("closed"):
            try:
                slab = _add_slab(model, e, owner_history, storey_placement, geom_ctx)
                if slab:
                    _add_property_set(model, slab, owner_history, "Concrete", DEFAULT_SLAB_HEIGHT)
                    contained_elements.append(slab)
            except Exception as ex:
                logger.debug(f"[ifc_gen] slab skip: {ex}")

    # ── Spaces ───────────────────────────────────────────────────────────────
    for e in bim_structure.get("spaces", []):
        if e["type"] == "polyline" and e.get("closed"):
            try:
                space = _add_space(model, e, owner_history, storey_placement, geom_ctx)
                if space:
                    contained_elements.append(space)
            except Exception as ex:
                logger.debug(f"[ifc_gen] space skip: {ex}")

    # ── Contain all elements in storey ───────────────────────────────────────
    if contained_elements:
        import ifcopenshell
        model.create_entity("IfcRelContainedInSpatialStructure",
                            GlobalId=ifcopenshell.guid.new(),
                            OwnerHistory=owner_history,
                            RelatingStructure=storey,
                            RelatedElements=contained_elements)

    # ── Export ───────────────────────────────────────────────────────────────
    fd, tmp = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "wb") as _:
            pass  # close fd
        model.write(tmp)
        with open(tmp, "rb") as f:
            ifc_bytes = f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    return ifc_bytes


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE — PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def convert_cad_to_ifc(filename: str, file_bytes: bytes) -> Tuple[bytes, dict]:
    """
    Full CAD → IFC pipeline. Returns (ifc_bytes, conversion_report).
    The conversion report contains stats for internal logging only.
    Raises RuntimeError if ifcopenshell is missing.
    """
    kind = detect_input_type(filename)
    logger.info(f"[cad_to_ifc] start | file={filename} | type={kind} | size={len(file_bytes)}")

    # Step 1 — Extract
    raw = extract_geometry(filename, file_bytes)
    logger.info(f"[cad_to_ifc] extracted {len(raw)} raw entities")

    # Step 2 — Normalize
    normalized = normalize_entities(raw)

    # Step 3 — Clean
    cleaned = clean_geometry(normalized)
    logger.info(f"[cad_to_ifc] cleaned → {len(cleaned)} entities")

    # Step 4 — Semantics
    semantic = infer_semantics(cleaned)

    # Step 5 — BIM structure
    bim = build_bim_structure(semantic)
    counts = {k: len(v) for k, v in bim.items()}
    logger.info(f"[cad_to_ifc] bim structure: {counts}")

    # Steps 6–9 — Generate IFC
    ifc_bytes = generate_ifc(bim)
    logger.info(f"[cad_to_ifc] generated IFC | {len(ifc_bytes)} bytes")

    report = {
        "source_type":     kind,
        "raw_entities":    len(raw),
        "cleaned_entities": len(cleaned),
        "bim_counts":      counts,
        "ifc_size_bytes":  len(ifc_bytes),
        "conversion_note": "CAD files may contain some inaccuracies in the converted IFC.",
    }

    return ifc_bytes, report
