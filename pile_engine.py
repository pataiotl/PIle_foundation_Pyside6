"""Headless engineering core for the PySide6 pile foundation app.

This module is mechanically split from the original Streamlit application so the
calculation, envelope, SAP2000 parsing, drawing, and report logic stay aligned
with the existing tool while the desktop UI lives separately.
"""
from __future__ import annotations

import io
import math
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

APP_TITLE = "Pile Foundation Designer"
APP_SUBTITLE = "ACI 318-25 style RC pile-cap design aid | 2 to 12 piles | SAP2000 reactions or manual input"


# =============================================================================
# CONSTANTS
# =============================================================================

BAR_DATABASE_MM = {
    "DB10": {"diameter": 10.0, "area": 78.5},
    "DB12": {"diameter": 12.0, "area": 113.1},
    "DB16": {"diameter": 16.0, "area": 201.1},
    "DB20": {"diameter": 20.0, "area": 314.2},
    "DB25": {"diameter": 25.0, "area": 490.9},
    "DB28": {"diameter": 28.0, "area": 615.8},
    "DB32": {"diameter": 32.0, "area": 804.2},
    "DB36": {"diameter": 36.0, "area": 1017.9},
    "DB40": {"diameter": 40.0, "area": 1256.6},
}

DEFAULT_STRENGTH_COMBOS = {
    "ULS-Manual": {
        "Pu_kN": 5000.0,
        "Mux_kNm": 0.0,
        "Muy_kNm": 0.0,
        "Vux_kN": 0.0,
        "Vuy_kN": 0.0,
        "Tuz_kNm": 0.0,
    }
}

ACI_REFERENCES = {
    "general": "ACI CODE-318-25 should be used directly for final design and detailing.",
    "flexure": "ACI-style strength design: phi Mn >= Mu. Rectangular section, singly-reinforced simplification.",
    "one_way_shear": "ACI-style concrete shear expression including rho_w and lambda_s size effect; verify exact ACI 318-25 modifiers.",
    "two_way_shear": "ACI-style two-way shear/punching check using the least of beta, alpha_s, and upper-limit equations.",
    "development": "ACI-style tension development estimate including cb/Ktr confinement term; engineer must verify Chapter 25 modifiers.",
    "stm": "MacGregor-style practical load-path/strut-and-tie advisory for deep pile caps; not a substitute for full STM.",
}

PILE_LAYOUT_TEMPLATE_VERSION = "reference-arrangements-2026-05"

GROUP_DEFAULT_COLUMNS = {
    "Group Name": "Foundation 1",
    "Joint IDs": "",
    "No. Piles": 4,
    "Thickness (mm)": 1200.0,
    "Pile Dia (mm)": 600.0,
    "Spacing X (mm)": 1800.0,
    "Spacing Y (mm)": 1800.0,
    "Edge (mm)": 600.0,
    "Pile Comp Cap (kN)": 1800.0,
    "Pile Tension Cap (kN)": 400.0,
}

MANUAL_SERVICE_LOAD_DEFAULTS = {
    "D Ps (kN)": 3000.0,
    "D Msx (kN-m)": 0.0,
    "D Msy (kN-m)": 0.0,
    "L Ps (kN)": 2000.0,
    "L Msx (kN-m)": 0.0,
    "L Msy (kN-m)": 0.0,
    "D Factor": 1.2,
    "L Factor": 1.6,
}

MANUAL_SERVICE_LOAD_ALIASES = {
    "D Ps (kN)": ["D Pu (kN)", "Pu (kN)"],
    "D Msx (kN-m)": ["D Mux (kN-m)", "Mux (kN-m)"],
    "D Msy (kN-m)": ["D Muy (kN-m)", "Muy (kN-m)"],
    "L Ps (kN)": ["L Pu (kN)"],
    "L Msx (kN-m)": ["L Mux (kN-m)"],
    "L Msy (kN-m)": ["L Muy (kN-m)"],
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Material:
    fc_MPa: float = 35.0
    fy_MPa: float = 500.0
    lambda_c: float = 1.0
    gamma_conc_kN_m3: float = 24.0
    phi_flexure: float = 0.90
    phi_shear: float = 0.75
    phi_bearing: float = 0.65
    phi_stm_tie: float = 0.75

@dataclass
class Geometry:
    n_piles: int = 4
    pile_diameter_mm: float = 600.0
    pile_capacity_comp_kN: float = 1800.0
    pile_capacity_tension_kN: float = 400.0
    cap_thickness_mm: float = 1200.0
    bottom_cover_mm: float = 100.0
    top_cover_mm: float = 75.0
    side_cover_mm: float = 75.0
    column_bx_mm: float = 800.0
    column_by_mm: float = 800.0
    pedestal_bx_mm: float = 800.0
    pedestal_by_mm: float = 800.0
    edge_from_pile_edge_mm: float = 600.0
    spacing_x_mm: float = 1800.0
    spacing_y_mm: float = 1800.0
    use_pedestal_for_shear: bool = True
    column_location: str = "Interior"

@dataclass
class Reinforcement:
    main_bar_x: str = "DB25"
    main_bar_y: str = "DB25"
    top_bar: str = "DB16"
    spacing_x_mm: float = 150.0
    spacing_y_mm: float = 150.0
    top_spacing_mm: float = 200.0
    side_face_bar: str = "DB16"
    side_face_spacing_mm: float = 250.0
    hook_extension_mm: float = 300.0
    preferred_spacing_step_mm: float = 25.0

@dataclass
class LoadCase:
    name: str = "ULS-Manual"
    Pu_kN: float = 5000.0
    Mux_kNm: float = 0.0
    Muy_kNm: float = 0.0
    Vux_kN: float = 0.0
    Vuy_kN: float = 0.0
    Tuz_kNm: float = 0.0

@dataclass
class PilePoint:
    label: str
    x_mm: float
    y_mm: float
    diameter_mm: float
    reaction_kN: float = 0.0

@dataclass
class CheckResult:
    name: str
    demand: float
    capacity: float
    ratio: float
    unit: str
    status: str
    note: str = ""

@dataclass
class DesignState:
    material: Material
    geometry: Geometry
    reinforcement: Reinforcement
    loadcase: LoadCase
    piles: List[PilePoint]
    cap_length_x_mm: float
    cap_width_y_mm: float
    effective_depth_x_mm: float
    effective_depth_y_mm: float
    self_weight_kN: float = 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: float, nd: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.{nd}f}"


def status_from_ratio(ratio: float, warn_limit: float = 0.95) -> str:
    if ratio <= warn_limit:
        return "PASS"
    if ratio <= 1.0:
        return "NEAR"
    return "FAIL"


def status_badge(status: str) -> str:
    s = status.upper()
    if s == "PASS":
        return "✅ PASS"
    if s == "NEAR":
        return "⚠️ NEAR"
    if s == "FAIL":
        return "❌ FAIL"
    if s == "STM":
        return "STM REVIEW"
    return s


def bar_area(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB25"])["area"]


def bar_diameter(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB25"])["diameter"]


def round_up_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


def round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def kN_to_N(value_kN: float) -> float:
    return value_kN * 1000.0


def kNm_to_Nmm(value_kNm: float) -> float:
    return value_kNm * 1_000_000.0


def Nmm_to_kNm(value_Nmm: float) -> float:
    return value_Nmm / 1_000_000.0


def kN_from_N(value_N: float) -> float:
    return value_N / 1000.0


def get_session_default(key: str, value: Any) -> Any:
    return value

# =============================================================================
# PILE LAYOUT GENERATION
# =============================================================================

def template_layout(n: int, sx: float, sy: float) -> List[Tuple[float, float]]:
    """
    Auto-generate pile coordinates around centroid using the reference
    arrangement sketches.

    Coordinates:
        x positive to right, y positive upward in plan.
        Units: mm.

    The typical spacing inputs represent the 3D center-to-center spacing shown
    in the reference sketches. Layouts with diagonal/triangular geometry derive
    their offsets from that spacing with sqrt(2) or sqrt(3) factors.
    """
    if n < 2 or n > 12:
        raise ValueError("n_piles must be between 2 and 12")

    rt2 = math.sqrt(2.0)
    rt3 = math.sqrt(3.0)

    # Reference templates from the attached pile arrangement sketches. User can
    # still edit the generated coordinates before running the design.
    templates: Dict[int, List[Tuple[float, float]]] = {
        # 2 piles: vertical pair.
        2: [(0.0, -0.5 * sy), (0.0, 0.5 * sy)],
        # 3 piles: equilateral triangle, centered at its centroid.
        3: [(-0.5 * sx, -rt3 * sy / 6.0), (0.5 * sx, -rt3 * sy / 6.0), (0.0, rt3 * sy / 3.0)],
        4: [(-0.5 * sx, -0.5 * sy), (0.5 * sx, -0.5 * sy), (-0.5 * sx, 0.5 * sy), (0.5 * sx, 0.5 * sy)],
        # 5 piles: four corner piles plus one center pile; corner spacing is 3*sqrt(2)D.
        5: [(-sx / rt2, -sy / rt2), (sx / rt2, -sy / rt2), (0.0, 0.0), (-sx / rt2, sy / rt2), (sx / rt2, sy / rt2)],
        6: [(-sx, -0.5 * sy), (0.0, -0.5 * sy), (sx, -0.5 * sy), (-sx, 0.5 * sy), (0.0, 0.5 * sy), (sx, 0.5 * sy)],
        # 7 piles: center pile plus six piles on a regular hexagonal ring.
        7: [(0.0, -sy), (-rt3 * sx / 2.0, -0.5 * sy), (rt3 * sx / 2.0, -0.5 * sy),
            (0.0, 0.0), (-rt3 * sx / 2.0, 0.5 * sy), (rt3 * sx / 2.0, 0.5 * sy), (0.0, sy)],
        # 8 piles: compact staggered 3-2-3 layout from the reference sketch.
        8: [(-rt2 * sx, -sy / rt2), (0.0, -sy / rt2), (rt2 * sx, -sy / rt2),
            (-sx / rt2, 0.0), (sx / rt2, 0.0),
            (-rt2 * sx, sy / rt2), (0.0, sy / rt2), (rt2 * sx, sy / rt2)],
        9: [(-sx, -sy), (0.0, -sy), (sx, -sy), (-sx, 0.0), (0.0, 0.0), (sx, 0.0), (-sx, sy), (0.0, sy), (sx, sy)],
        # 10 and 11 piles: staggered triangular-grid layouts with total row
        # spacing of 3*sqrt(3)D between exterior rows.
        10: [(-sx, -rt3 * sy / 2.0), (0.0, -rt3 * sy / 2.0), (sx, -rt3 * sy / 2.0),
             (-1.5 * sx, 0.0), (-0.5 * sx, 0.0), (0.5 * sx, 0.0), (1.5 * sx, 0.0),
             (-sx, rt3 * sy / 2.0), (0.0, rt3 * sy / 2.0), (sx, rt3 * sy / 2.0)],
        11: [(-1.5 * sx, -rt3 * sy / 2.0), (-0.5 * sx, -rt3 * sy / 2.0), (0.5 * sx, -rt3 * sy / 2.0), (1.5 * sx, -rt3 * sy / 2.0),
             (-sx, 0.0), (0.0, 0.0), (sx, 0.0),
             (-1.5 * sx, rt3 * sy / 2.0), (-0.5 * sx, rt3 * sy / 2.0), (0.5 * sx, rt3 * sy / 2.0), (1.5 * sx, rt3 * sy / 2.0)],
        12: [(-1.5 * sx, -sy), (-0.5 * sx, -sy), (0.5 * sx, -sy), (1.5 * sx, -sy),
             (-1.5 * sx, 0.0), (-0.5 * sx, 0.0), (0.5 * sx, 0.0), (1.5 * sx, 0.0),
             (-1.5 * sx, sy), (-0.5 * sx, sy), (0.5 * sx, sy), (1.5 * sx, sy)],
    }
    return templates[n]


def make_piles(n: int, sx: float, sy: float, pile_dia: float) -> List[PilePoint]:
    coords = template_layout(n, sx, sy)
    piles = []
    for i, (x, y) in enumerate(coords, start=1):
        piles.append(PilePoint(label=f"P{i}", x_mm=float(x), y_mm=float(y), diameter_mm=float(pile_dia)))
    return piles


def normalize_pile_centroid(piles: List[PilePoint]) -> List[PilePoint]:
    if not piles:
        return piles
    xbar = np.mean([p.x_mm for p in piles])
    ybar = np.mean([p.y_mm for p in piles])
    out = []
    for p in piles:
        out.append(PilePoint(p.label, p.x_mm - xbar, p.y_mm - ybar, p.diameter_mm, p.reaction_kN))
    return out


def piles_to_dataframe(piles: List[PilePoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pile": p.label,
                "x_mm": p.x_mm,
                "y_mm": p.y_mm,
                "diameter_mm": p.diameter_mm,
                "reaction_kN": p.reaction_kN,
            }
            for p in piles
        ]
    )


def dataframe_to_piles(df: pd.DataFrame, default_dia: float) -> List[PilePoint]:
    piles = []
    for i, row in df.iterrows():
        label = str(row.get("Pile", f"P{i + 1}"))
        x = safe_float(row.get("x_mm", 0.0))
        y = safe_float(row.get("y_mm", 0.0))
        dia = safe_float(row.get("diameter_mm", default_dia), default_dia)
        r = safe_float(row.get("reaction_kN", 0.0))
        piles.append(PilePoint(label=label, x_mm=x, y_mm=y, diameter_mm=dia, reaction_kN=r))
    return normalize_pile_centroid(piles)


def cap_dimensions_from_piles(piles: List[PilePoint], edge_mm: float) -> Tuple[float, float]:
    if not piles:
        return 0.0, 0.0
    min_x = min(p.x_mm - p.diameter_mm / 2 for p in piles) - edge_mm
    max_x = max(p.x_mm + p.diameter_mm / 2 for p in piles) + edge_mm
    min_y = min(p.y_mm - p.diameter_mm / 2 for p in piles) - edge_mm
    max_y = max(p.y_mm + p.diameter_mm / 2 for p in piles) + edge_mm
    return max_x - min_x, max_y - min_y


# =============================================================================
# LOAD DISTRIBUTION
# =============================================================================

def distribute_vertical_load_to_piles(
    piles: List[PilePoint],
    Pu_kN: float,
    Mux_kNm: float,
    Muy_kNm: float,
) -> List[PilePoint]:
    """
    Elastic rigid pile-cap distribution.

    Formula:
        R_i = P/n + Mx*y_i/sum(y²) + My*x_i/sum(x²)

    Notes:
        - x, y in m.
        - Mx is moment about X axis causing gradient along y.
        - My is moment about Y axis causing gradient along x.
        - Positive R means compression on pile.
        - Apply the desired sign convention before calling this function.
    """
    if not piles:
        return []
    n = len(piles)
    P_total = Pu_kN
    xs_m = np.array([mm_to_m(p.x_mm) for p in piles], dtype=float)
    ys_m = np.array([mm_to_m(p.y_mm) for p in piles], dtype=float)
    sum_x2 = float(np.sum(xs_m**2))
    sum_y2 = float(np.sum(ys_m**2))

    base = P_total / n
    reactions = np.full(n, base, dtype=float)

    if sum_y2 > 1e-12:
        reactions += Mux_kNm * ys_m / sum_y2
    if sum_x2 > 1e-12:
        reactions += Muy_kNm * xs_m / sum_x2

    out = []
    for p, r in zip(piles, reactions):
        out.append(PilePoint(p.label, p.x_mm, p.y_mm, p.diameter_mm, float(r)))
    return out


def pile_group_result_table(piles: List[PilePoint], geom: Geometry) -> pd.DataFrame:
    rows = []
    for p in piles:
        comp_ratio = max(p.reaction_kN, 0) / max(geom.pile_capacity_comp_kN, 1e-9)
        tension = max(-p.reaction_kN, 0)
        tension_ratio = tension / max(geom.pile_capacity_tension_kN, 1e-9)
        rows.append(
            {
                "Pile": p.label,
                "No. Piles": geom.n_piles,
                "x (mm)": p.x_mm,
                "y (mm)": p.y_mm,
                "R, compression + (kN)": p.reaction_kN,
                "Compression demand (kN)": max(p.reaction_kN, 0.0),
                "Uplift demand (kN)": tension,
                "Compression ratio": comp_ratio,
                "Tension ratio": tension_ratio,
                "Status": status_badge("FAIL" if max(comp_ratio, tension_ratio) > 1.0 else "PASS"),
            }
        )
    return pd.DataFrame(rows)

# =============================================================================
# STATE IMPORT / EXPORT
# =============================================================================

def key_value_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Key": key, "Value": value} for key, value in data.items()])


def dataframe_to_key_values(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "Key" not in df.columns or "Value" not in df.columns:
        return {}
    return {str(row["Key"]): row["Value"] for _, row in df.iterrows() if str(row.get("Key", "")).strip()}


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def make_saved_state_xlsx(state: DesignState, results: Dict[str, Any], include_self_weight: bool) -> bytes:
    raise RuntimeError("Use the PySide6 override defined at the end of this module.")

def read_saved_state_xlsx(uploaded_file) -> Optional[Dict[str, Any]]:
    raise RuntimeError("Use the PySide6 override defined at the end of this module.")

_SAVED_STATE_PAYLOAD: Dict[str, Any] = {}

def value_from_saved(section: str, key: str, default: Any) -> Any:
    value = _SAVED_STATE_PAYLOAD.get(section, {}).get(key, default)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value

def saved_float(section: str, key: str, default: float) -> float:
    return safe_float(value_from_saved(section, key, default), default)

def saved_int(section: str, key: str, default: int) -> int:
    return int(round(saved_float(section, key, float(default))))

def saved_bool(section: str, key: str, default: bool) -> bool:
    return coerce_bool(value_from_saved(section, key, default), default)

def saved_choice(section: str, key: str, default: str, choices: List[str]) -> str:
    value = str(value_from_saved(section, key, default))
    return value if value in choices else default

def apply_saved_pile_layout_if_needed(default_dia: float) -> None:
    return None

# =============================================================================
# RC DESIGN FORMULAS
# =============================================================================

def beta1_aci(fc_MPa: float) -> float:
    """
    ACI-style beta1 expression in MPa.
    SI approximation of the ACI beta1 table: 28 MPa is approximately 4000 psi;
    beta1 decreases 0.05 for each additional 7 MPa to a minimum of 0.65.
    Verify against the governing code edition.
    """
    if fc_MPa <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * ((fc_MPa - 28.0) / 7.0))


def flexural_As_required(
    Mu_kNm: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    """
    Required tensile steel area for singly reinforced rectangular section.
    Minimum steel uses the common ACI beam expression as a preliminary design
    floor; verify footing/deep-beam minimum reinforcement provisions for final
    pile-cap classification.

    Solves:
        phi * As * fy * (d - a/2) >= Mu
        a = As * fy / (0.85 fc b)

    Units:
        Mu: kN-m
        b, d: mm
        fc, fy: MPa = N/mm²
        As: mm²
    """
    Mu_Nmm = abs(kNm_to_Nmm(Mu_kNm))
    if Mu_Nmm <= 1e-9:
        As_min = max(0.25 * math.sqrt(fc_MPa) / fy_MPa, 1.4 / fy_MPa) * b_mm * d_mm
        return {
            "As_req_mm2": As_min,
            "As_strength_mm2": 0.0,
            "As_min_mm2": As_min,
            "rho": As_min / (b_mm * d_mm),
            "a_mm": 0.0,
            "phiMn_kNm": 0.0,
        }

    A = phi * fy_MPa * fy_MPa / (2.0 * 0.85 * fc_MPa * b_mm)
    B = -phi * fy_MPa * d_mm
    C = Mu_Nmm

    disc = B * B - 4.0 * A * C
    if disc < 0:
        # Section is too small or heavily reinforced. Return high number as signal.
        As_strength = np.nan
    else:
        root1 = (-B - math.sqrt(disc)) / (2.0 * A)
        root2 = (-B + math.sqrt(disc)) / (2.0 * A)
        candidates = [r for r in (root1, root2) if r > 0]
        As_strength = min(candidates) if candidates else np.nan

    As_min = max(0.25 * math.sqrt(fc_MPa) / fy_MPa, 1.4 / fy_MPa) * b_mm * d_mm
    As_req = max(As_min, As_strength if np.isfinite(As_strength) else 1e99)
    a = As_req * fy_MPa / (0.85 * fc_MPa * b_mm)
    phiMn = phi * As_req * fy_MPa * (d_mm - a / 2.0)
    return {
        "As_req_mm2": As_req,
        "As_strength_mm2": As_strength,
        "As_min_mm2": As_min,
        "rho": As_req / (b_mm * d_mm),
        "a_mm": a,
        "phiMn_kNm": Nmm_to_kNm(phiMn),
    }


def flexural_capacity(
    As_mm2: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    a = As_mm2 * fy_MPa / (0.85 * fc_MPa * b_mm)
    Mn_Nmm = As_mm2 * fy_MPa * max(d_mm - a / 2.0, 0.0)
    phiMn_Nmm = phi * Mn_Nmm
    c = a / max(beta1_aci(fc_MPa), 1e-9)
    eps_t = 0.003 * (d_mm - c) / max(c, 1e-9)
    return {
        "a_mm": a,
        "c_mm": c,
        "eps_t": eps_t,
        "Mn_kNm": Nmm_to_kNm(Mn_Nmm),
        "phiMn_kNm": Nmm_to_kNm(phiMn_Nmm),
    }


def one_way_shear_capacity(
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float,
    phi: float = 0.75,
    rho_w: float = 0.0025,
    Nu_kN: float = 0.0,
    Ag_mm2: Optional[float] = None,
    use_close_pile_aci318_25: bool = False,
) -> Dict[str, float]:
    """
    ACI-style one-way concrete shear for members without shear reinforcement:
        Vc = [0.66 * lambda_s * lambda * rho_w^(1/3) * sqrt(fc') + Nu/(6Ag)] * b * d

    A conservative cap is also applied to the concrete stress term:
        vc <= 0.42 * lambda * sqrt(fc')

    SI units: fc in MPa, b and d in mm. The lambda_s expression uses d in mm.
    """
    lambda_s_general = min(1.0, math.sqrt(2.0 / (1.0 + 0.004 * max(d_mm, 0.0))))
    lambda_s = 1.0 if use_close_pile_aci318_25 else lambda_s_general
    rho_use = min(max(rho_w, 0.0001), 0.08)
    axial_stress = 0.0
    if Ag_mm2 and Ag_mm2 > 0:
        axial_stress = kN_to_N(Nu_kN) / (6.0 * Ag_mm2)
        axial_stress = min(axial_stress, 0.05 * fc_MPa)
    if use_close_pile_aci318_25:
        # SI equivalent of the restored ACI 318-14/IP expression 2 lambda sqrt(fc') bw d.
        vc_refined = 0.17 * lambda_c * math.sqrt(fc_MPa)
        vc_limit = vc_refined
        vc_use = vc_refined
        formula = "ACI 318-25 close-pile one-way shear"
    else:
        vc_refined = 0.66 * lambda_s * lambda_c * (rho_use ** (1.0 / 3.0)) * math.sqrt(fc_MPa) + axial_stress
        vc_limit = 0.42 * lambda_c * math.sqrt(fc_MPa)
        vc_use = min(vc_refined, vc_limit)
        formula = "ACI-style general one-way shear"
    Vc_N = vc_use * b_mm * d_mm
    return {
        "Vc_kN": kN_from_N(Vc_N),
        "phiVc_kN": kN_from_N(phi * Vc_N),
        "lambda_s": lambda_s,
        "lambda_s_general": lambda_s_general,
        "rho_w": rho_use,
        "vc_MPa": vc_use,
        "vc_refined_MPa": vc_refined,
        "vc_limit_MPa": vc_limit,
        "Nu_over_6Ag_MPa": axial_stress,
        "use_close_pile_aci318_25": use_close_pile_aci318_25,
        "formula": formula,
    }


def two_way_shear_capacity(
    bo_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float,
    phi: float = 0.75,
    loaded_bx_mm: float = 1.0,
    loaded_by_mm: float = 1.0,
    column_location: str = "Interior",
    use_close_pile_aci318_25: bool = False,
) -> Dict[str, float]:
    """
    ACI-style two-way punching shear without shear reinforcement:
        vc is the least of:
        a) 0.33 * lambda_s * lambda * sqrt(fc')
        b) (0.17 + 0.33 / beta) * lambda_s * lambda * sqrt(fc')
        c) (0.17 + 0.083 * alpha_s * d / bo) * lambda_s * lambda * sqrt(fc')

    SI units: fc in MPa, bo and d in mm. The lambda_s expression uses d in mm.
    """
    lambda_s_general = min(1.0, math.sqrt(2.0 / (1.0 + 0.004 * max(d_mm, 0.0))))
    lambda_s = 1.0 if use_close_pile_aci318_25 else lambda_s_general
    loaded_short = max(min(loaded_bx_mm, loaded_by_mm), 1e-9)
    beta = max(max(loaded_bx_mm, loaded_by_mm) / loaded_short, 1.0)
    loc = str(column_location or "Interior").strip().lower()
    alpha_s = 40.0 if loc.startswith("interior") else 30.0 if loc.startswith("edge") else 20.0
    root_fc = math.sqrt(fc_MPa)
    vc_a = 0.33 * lambda_s * lambda_c * root_fc
    vc_b = (0.17 + 0.33 / beta) * lambda_s * lambda_c * root_fc
    vc_c = (0.17 + 0.083 * alpha_s * d_mm / max(bo_mm, 1e-9)) * lambda_s * lambda_c * root_fc
    options = {"upper_limit": vc_a, "beta_limit": vc_b, "alpha_s_limit": vc_c}
    governing = min(options, key=options.get)
    vc_use = options[governing]
    Vc_N = vc_use * bo_mm * d_mm
    return {
        "Vc_kN": kN_from_N(Vc_N),
        "phiVc_kN": kN_from_N(phi * Vc_N),
        "lambda_s": lambda_s,
        "lambda_s_general": lambda_s_general,
        "beta": beta,
        "alpha_s": alpha_s,
        "vc_MPa": vc_use,
        "vc_upper_limit_MPa": vc_a,
        "vc_beta_limit_MPa": vc_b,
        "vc_alpha_s_limit_MPa": vc_c,
        "governing_equation": governing,
        "use_close_pile_aci318_25": use_close_pile_aci318_25,
    }


def concrete_bearing_capacity(
    area_loaded_mm2: float,
    fc_MPa: float,
    phi: float = 0.65,
    bearing_factor: float = 0.85,
    area_ratio_factor: float = 1.0,
) -> Dict[str, float]:
    Pn_N = bearing_factor * fc_MPa * area_loaded_mm2 * area_ratio_factor
    return {"Pn_kN": kN_from_N(Pn_N), "phiPn_kN": kN_from_N(phi * Pn_N)}


def development_length_tension_estimate(
    db_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    lambda_c: float = 1.0,
    psi_t: float = 1.0,
    psi_e: float = 1.0,
    psi_s: float = 1.0,
    clear_cover_mm: Optional[float] = None,
    clear_spacing_mm: Optional[float] = None,
    Ktr_mm: float = 0.0,
    min_ld_mm: float = 300.0,
) -> Dict[str, float]:
    """
    Editable ACI-style development length estimate using a cb/Ktr confinement term.

    This remains a design aid: exact Chapter 25 modifiers, excess reinforcement,
    epoxy/coating, casting position, hooks, headed bars, and transverse steel
    detailing must be verified for final design.
    """
    cover_cb = clear_cover_mm + 0.5 * db_mm if clear_cover_mm is not None else db_mm
    spacing_cb = 0.5 * clear_spacing_mm if clear_spacing_mm is not None else db_mm
    cb = max(min(cover_cb, spacing_cb), 0.5 * db_mm)
    confinement_factor = min(max((cb + max(Ktr_mm, 0.0)) / max(db_mm, 1e-9), 1.5), 2.5)
    ld_calc = (fy_MPa * psi_t * psi_e * psi_s / (1.1 * lambda_c * math.sqrt(fc_MPa))) * db_mm / confinement_factor
    ld = max(min_ld_mm, ld_calc)
    return {
        "ld_mm": ld,
        "ld_calc_mm": ld_calc,
        "cb_mm": cb,
        "Ktr_mm": max(Ktr_mm, 0.0),
        "confinement_factor": confinement_factor,
    }


def spacing_for_As(
    As_req_mm2: float,
    bar: str,
    strip_width_mm: float,
    preferred_step_mm: float = 25.0,
    s_min_mm: float = 75.0,
    s_max_mm: float = 300.0,
) -> Dict[str, float]:
    """
    Convert required total steel area across a strip width to a spacing.
    """
    Ab = bar_area(bar)
    if As_req_mm2 <= 0:
        return {"spacing_req_mm": s_max_mm, "spacing_use_mm": s_max_mm, "n_bars": 0, "As_prov_mm2": 0.0}
    s_req = Ab * strip_width_mm / As_req_mm2
    s_use = min(s_max_mm, max(s_min_mm, round_down_to_step(s_req, preferred_step_mm)))
    n_bars = int(math.floor(strip_width_mm / s_use)) + 1
    As_prov = n_bars * Ab
    return {"spacing_req_mm": s_req, "spacing_use_mm": s_use, "n_bars": n_bars, "As_prov_mm2": As_prov}


def provided_spacing_As(bar: str, spacing_mm: float, strip_width_mm: float) -> Dict[str, float]:
    spacing_use = max(safe_float(spacing_mm, 150.0), 1.0)
    n_bars = int(math.floor(strip_width_mm / spacing_use)) + 1
    As_prov = n_bars * bar_area(bar)
    return {
        "spacing_req_mm": spacing_use,
        "spacing_use_mm": spacing_use,
        "n_bars": n_bars,
        "As_prov_mm2": As_prov,
    }


# =============================================================================
# FOOTING / PILE CAP DESIGN ACTIONS
# =============================================================================

def effective_depths(geom: Geometry, reinf: Reinforcement) -> Tuple[float, float]:
    """
    Effective depth to bottom reinforcement in each orthogonal layer.
    Assume X-direction bars are the lower layer, Y-direction bars above them.
    """
    db_x = bar_diameter(reinf.main_bar_x)
    db_y = bar_diameter(reinf.main_bar_y)
    d_x = geom.cap_thickness_mm - geom.bottom_cover_mm - db_x / 2.0
    d_y = geom.cap_thickness_mm - geom.bottom_cover_mm - db_x - db_y / 2.0
    return max(d_x, 1.0), max(d_y, 1.0)


def top_effective_depths(geom: Geometry, reinf: Reinforcement) -> Tuple[float, float]:
    """
    Effective depths for top reinforcement checked with bottom face in compression.
    Top X and Y layers use the same top bar size; Y is treated as the second layer.
    """
    db = bar_diameter(reinf.top_bar)
    d_top_x = geom.cap_thickness_mm - geom.top_cover_mm - db / 2.0
    d_top_y = geom.cap_thickness_mm - geom.top_cover_mm - db - db / 2.0
    return max(d_top_x, 1.0), max(d_top_y, 1.0)


def close_pile_spacing_check(piles: List[PilePoint], geom: Geometry, limit_factor: float = 4.0) -> Dict[str, Any]:
    if len(piles) < 2:
        return {
            "applies": False,
            "limit_factor": limit_factor,
            "limit_mm": limit_factor * geom.pile_diameter_mm,
            "max_nearest_spacing_mm": 0.0,
            "nearest_spacings_mm": {},
            "note": "Close-pile shear provision needs at least two piles.",
        }

    nearest = {}
    for p in piles:
        distances = [
            math.hypot(p.x_mm - other.x_mm, p.y_mm - other.y_mm)
            for other in piles
            if other.label != p.label
        ]
        nearest[p.label] = min(distances) if distances else 0.0

    max_nearest = max(nearest.values()) if nearest else 0.0
    limit = limit_factor * geom.pile_diameter_mm
    applies = max_nearest <= limit + 1e-9
    return {
        "applies": applies,
        "limit_factor": limit_factor,
        "limit_mm": limit,
        "max_nearest_spacing_mm": max_nearest,
        "nearest_spacings_mm": nearest,
        "note": (
            f"ACI 318-25 close-pile shear provision applies: max nearest pile spacing {max_nearest:.0f} mm <= {limit_factor:.0f}D = {limit:.0f} mm."
            if applies
            else f"ACI 318-25 close-pile shear provision does not apply: max nearest pile spacing {max_nearest:.0f} mm > {limit_factor:.0f}D = {limit:.0f} mm."
        ),
    }


def calculate_material_takeoff(state: DesignState, results: Dict[str, Any]) -> Dict[str, float]:
    """Estimates concrete volume and main flexural rebar weight."""
    vol_m3 = mm_to_m(state.cap_length_x_mm) * mm_to_m(state.cap_width_y_mm) * mm_to_m(state.geometry.cap_thickness_mm)
    density_kg_mm3 = 7.85e-6

    weight_x_kg = results["spacing_x"]["As_prov_mm2"] * state.cap_length_x_mm * density_kg_mm3
    weight_y_kg = results["spacing_y"]["As_prov_mm2"] * state.cap_width_y_mm * density_kg_mm3
    weight_top_x_kg = results.get("top_spacing_x", {}).get("As_prov_mm2", 0.0) * state.cap_length_x_mm * density_kg_mm3
    weight_top_y_kg = results.get("top_spacing_y", {}).get("As_prov_mm2", 0.0) * state.cap_width_y_mm * density_kg_mm3
    total_rebar_kg = weight_x_kg + weight_y_kg + weight_top_x_kg + weight_top_y_kg

    return {
        "concrete_vol_m3": vol_m3,
        "main_rebar_kg": total_rebar_kg,
        "rebar_ratio_kg_m3": total_rebar_kg / max(vol_m3, 1e-9)
    }


def self_weight_cap_kN(cap_x_mm: float, cap_y_mm: float, thickness_mm: float, gamma_kN_m3: float) -> float:
    vol_m3 = mm_to_m(cap_x_mm) * mm_to_m(cap_y_mm) * mm_to_m(thickness_mm)
    return vol_m3 * gamma_kN_m3


def design_moments_from_pile_reactions(
    piles: List[PilePoint],
    geom: Geometry,
    cap_x_mm: float,
    cap_y_mm: float,
) -> Dict[str, float]:
    """
    Simplified cantilever sectional moments at face of column/pedestal.

    For each direction, take pile reactions outside the loaded area face and
    multiply by lever arm to the face. The max of left/right or top/bottom is used.

    Mx-design here means bending requiring bars parallel to X axis, produced by
    cantilever action in Y direction. My-design means bars parallel to Y axis,
    produced by cantilever action in X direction.

    We report:
        M_for_X_bars: moments from piles above/below y face -> reinforcement along X.
        M_for_Y_bars: moments from piles left/right x face -> reinforcement along Y.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_face = loaded_bx / 2.0
    y_face = loaded_by / 2.0

    M_right = sum(max(p.reaction_kN, 0.0) * max(p.x_mm - x_face, 0.0) / 1000.0 for p in piles)
    M_left = sum(max(p.reaction_kN, 0.0) * max(-p.x_mm - x_face, 0.0) / 1000.0 for p in piles)

    M_top = sum(max(p.reaction_kN, 0.0) * max(p.y_mm - y_face, 0.0) / 1000.0 for p in piles)
    M_bottom = sum(max(p.reaction_kN, 0.0) * max(-p.y_mm - y_face, 0.0) / 1000.0 for p in piles)

    return {
        "M_right_for_Y_bars_kNm": M_right,
        "M_left_for_Y_bars_kNm": M_left,
        "M_top_for_X_bars_kNm": M_top,
        "M_bottom_for_X_bars_kNm": M_bottom,
        "M_for_Y_bars_kNm": max(M_right, M_left),
        "M_for_X_bars_kNm": max(M_top, M_bottom),
    }


def top_moments_from_tension_pile_reactions(
    piles: List[PilePoint],
    geom: Geometry,
) -> Dict[str, float]:
    """
    Top flexural demand from actual net pile uplift/tension.

    Compression-positive pile reactions push up on the cap and are handled by
    the bottom cantilever moment check. Negative net reactions mean the pile is
    in tension after axial force and biaxial moment have been combined. That
    downward pull on the cap creates top tension at the loaded-area face, so top
    bars are checked from net T_pile times lever arm.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_face = loaded_bx / 2.0
    y_face = loaded_by / 2.0

    tension_components = {p.label: max(-p.reaction_kN, 0.0) for p in piles}

    M_right = sum(tension_components[p.label] * max(p.x_mm - x_face, 0.0) / 1000.0 for p in piles)
    M_left = sum(tension_components[p.label] * max(-p.x_mm - x_face, 0.0) / 1000.0 for p in piles)
    M_top = sum(tension_components[p.label] * max(p.y_mm - y_face, 0.0) / 1000.0 for p in piles)
    M_bottom = sum(tension_components[p.label] * max(-p.y_mm - y_face, 0.0) / 1000.0 for p in piles)

    return {
        "M_right_for_Y_bars_kNm": M_right,
        "M_left_for_Y_bars_kNm": M_left,
        "M_top_for_X_bars_kNm": M_top,
        "M_bottom_for_X_bars_kNm": M_bottom,
        "M_for_Y_bars_kNm": max(M_right, M_left),
        "M_for_X_bars_kNm": max(M_top, M_bottom),
        "max_top_design_tension_component_kN": max(list(tension_components.values()) + [0.0]),
        "tension_components_kN": tension_components,
        "max_net_uplift_pile_kN": max([max(-p.reaction_kN, 0.0) for p in piles] + [0.0]),
    }


def governing_top_flexure_demand(
    continuous: Dict[str, Any],
    uplift_moment_kNm: float,
) -> Dict[str, Any]:
    continuous_moment = float(continuous.get("max_negative_kNm", 0.0))
    demand = max(continuous_moment, uplift_moment_kNm)
    if uplift_moment_kNm > continuous_moment + 1e-9:
        source = "net_uplift_tension_pile"
        note = "Top flexure governed by net pile tension/uplift force times lever arm to loaded-area face."
    elif continuous_moment > 1e-9:
        source = "continuous_negative_moment"
        note = str(continuous.get("note", "Continuous beam-strip negative support moment."))
    else:
        source = "nominal"
        note = str(continuous.get("note", "No top flexural demand from this load case."))
    return {
        "demand_kNm": demand,
        "continuous_negative_kNm": continuous_moment,
        "uplift_tension_kNm": uplift_moment_kNm,
        "source": source,
        "note": note,
    }


def continuous_beam_strip_moments(piles: List[PilePoint], axis: str) -> Dict[str, Any]:
    """
    Continuous beam-strip check over pile-line supports using matrix stiffness.

    Pile heads are vertical supports with free rotation. This does not add pile
    head fixity; it only captures negative cap bending over interior pile lines.
    A uniform equivalent downward load is used over the strip length and scaled
    to the sum of compression pile reactions.
    """
    axis = axis.lower()
    coords_raw = [p.x_mm if axis == "x" else p.y_mm for p in piles]
    support_coords = sorted({round(float(c), 6) for c in coords_raw})
    total_reaction = sum(max(p.reaction_kN, 0.0) for p in piles)
    if len(support_coords) < 3 or total_reaction <= 1e-9:
        return {
            "support_count": len(support_coords),
            "supports_mm": support_coords,
            "equivalent_w_kN_per_m": 0.0,
            "max_negative_kNm": 0.0,
            "max_positive_kNm": 0.0,
            "support_moments_kNm": {},
            "span_end_moments_kNm": [],
            "note": "No interior pile line; top flexure is nominal for this beam-strip direction.",
        }

    span_total_mm = support_coords[-1] - support_coords[0]
    if span_total_mm <= 1e-9:
        return {
            "support_count": len(support_coords),
            "supports_mm": support_coords,
            "equivalent_w_kN_per_m": 0.0,
            "max_negative_kNm": 0.0,
            "max_positive_kNm": 0.0,
            "support_moments_kNm": {},
            "span_end_moments_kNm": [],
            "note": "Pile-line span is zero; top flexure is nominal for this beam-strip direction.",
        }

    q_kN_per_mm = total_reaction / span_total_mm
    n = len(support_coords)
    ndof = 2 * n
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    for e, (x1, x2) in enumerate(zip(support_coords[:-1], support_coords[1:])):
        L = x2 - x1
        if L <= 1e-9:
            continue
        k = (1.0 / (L ** 3)) * np.array(
            [
                [12.0, 6.0 * L, -12.0, 6.0 * L],
                [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2],
                [-12.0, -6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2],
            ],
            dtype=float,
        )
        f = np.array(
            [
                -q_kN_per_mm * L / 2.0,
                -q_kN_per_mm * L ** 2 / 12.0,
                -q_kN_per_mm * L / 2.0,
                q_kN_per_mm * L ** 2 / 12.0,
            ],
            dtype=float,
        )
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for i, a in enumerate(dofs):
            F[a] += f[i]
            for j, b in enumerate(dofs):
                K[a, b] += k[i, j]

    free = [2 * i + 1 for i in range(n)]
    disp = np.zeros(ndof)
    try:
        disp[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    except np.linalg.LinAlgError:
        return {
            "support_count": len(support_coords),
            "supports_mm": support_coords,
            "equivalent_w_kN_per_m": q_kN_per_mm * 1000.0,
            "max_negative_kNm": 0.0,
            "max_positive_kNm": 0.0,
            "support_moments_kNm": {},
            "span_end_moments_kNm": [],
            "note": "Continuous beam-strip solve failed; review pile-line geometry.",
        }

    support_moments = {coord: [] for coord in support_coords}
    span_end_moments = []
    max_positive_kNmm = 0.0
    for e, (x1, x2) in enumerate(zip(support_coords[:-1], support_coords[1:])):
        L = x2 - x1
        k = (1.0 / (L ** 3)) * np.array(
            [
                [12.0, 6.0 * L, -12.0, 6.0 * L],
                [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2],
                [-12.0, -6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2],
            ],
            dtype=float,
        )
        f = np.array(
            [
                -q_kN_per_mm * L / 2.0,
                -q_kN_per_mm * L ** 2 / 12.0,
                -q_kN_per_mm * L / 2.0,
                q_kN_per_mm * L ** 2 / 12.0,
            ],
            dtype=float,
        )
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        end_forces = k @ disp[dofs] - f
        m_left = float(end_forces[1])
        m_right = float(end_forces[3])
        support_moments[x1].append(m_left)
        support_moments[x2].append(m_right)
        span_end_moments.append({"from_mm": x1, "to_mm": x2, "M_left_kNm": m_left / 1000.0, "M_right_kNm": m_right / 1000.0})

        r_left = q_kN_per_mm * L / 2.0 - (m_left + m_right) / L
        x_zero_shear = r_left / max(q_kN_per_mm, 1e-9)
        if 0.0 <= x_zero_shear <= L:
            m_pos = m_left + r_left * x_zero_shear - q_kN_per_mm * x_zero_shear ** 2 / 2.0
            max_positive_kNmm = max(max_positive_kNmm, m_pos)

    interior_coords = support_coords[1:-1]
    interior_moments = [m for coord in interior_coords for m in support_moments.get(coord, [])]
    max_negative_kNm = max([abs(m) / 1000.0 for m in interior_moments] + [0.0])
    support_moment_summary = {f"{coord:.0f}": max([abs(m) / 1000.0 for m in vals] + [0.0]) for coord, vals in support_moments.items()}

    return {
        "support_count": len(support_coords),
        "supports_mm": support_coords,
        "equivalent_w_kN_per_m": q_kN_per_mm * 1000.0,
        "max_negative_kNm": max_negative_kNm,
        "max_positive_kNm": max_positive_kNmm / 1000.0,
        "support_moments_kNm": support_moment_summary,
        "span_end_moments_kNm": span_end_moments,
        "note": "Continuous beam strip over vertical pile-line supports; pile-head rotations are free.",
    }


def one_way_shear_demands(
    piles: List[PilePoint],
    geom: Geometry,
    d_x_mm: float,
    d_y_mm: float,
) -> Dict[str, float]:
    """
    One-way shear at d from face of loaded area in both directions.

    Section normal to Y for X bars uses y-face + d_x.
    Section normal to X for Y bars uses x-face + d_y.

    We return maximum of two sides.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_sec = loaded_bx / 2.0 + d_y_mm
    y_sec = loaded_by / 2.0 + d_x_mm

    def one_way_contribution(coord: float, section: float, dia: float, positive_side: bool) -> float:
        distance = coord - section if positive_side else -coord - section
        if distance >= dia / 2.0:
            return 1.0
        if distance <= -dia / 2.0:
            return 0.0
        return 0.5

    V_right = sum(max(p.reaction_kN, 0.0) * one_way_contribution(p.x_mm, x_sec, p.diameter_mm, True) for p in piles)
    V_left = sum(max(p.reaction_kN, 0.0) * one_way_contribution(p.x_mm, x_sec, p.diameter_mm, False) for p in piles)
    V_top = sum(max(p.reaction_kN, 0.0) * one_way_contribution(p.y_mm, y_sec, p.diameter_mm, True) for p in piles)
    V_bottom = sum(max(p.reaction_kN, 0.0) * one_way_contribution(p.y_mm, y_sec, p.diameter_mm, False) for p in piles)

    return {
        "V_right_kN": V_right,
        "V_left_kN": V_left,
        "V_top_kN": V_top,
        "V_bottom_kN": V_bottom,
        "V_for_Y_bars_direction_kN": max(V_right, V_left),
        "V_for_X_bars_direction_kN": max(V_top, V_bottom),
    }


def punching_shear_demand(
    piles: List[PilePoint],
    geom: Geometry,
    d_avg_mm: float,
    Pu_total_kN: float,
) -> Dict[str, float]:
    """
    Simplified punching shear around loaded area at d/2.

    In pile caps, pile reactions inside the critical perimeter reduce punching demand.
    Pile reactions near the critical section are linearly proportioned across a
    band equal to one pile radius on each side of the critical section.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_lim = loaded_bx / 2.0 + d_avg_mm / 2.0
    y_lim = loaded_by / 2.0 + d_avg_mm / 2.0

    def signed_distance_to_rectangle(x_mm: float, y_mm: float) -> float:
        dx_out = max(abs(x_mm) - x_lim, 0.0)
        dy_out = max(abs(y_mm) - y_lim, 0.0)
        if dx_out > 0.0 or dy_out > 0.0:
            return math.hypot(dx_out, dy_out)
        return -min(x_lim - abs(x_mm), y_lim - abs(y_mm))

    reduction_rows = []
    for p in piles:
        radius = max(p.diameter_mm / 2.0, 1e-9)
        signed_dist = signed_distance_to_rectangle(p.x_mm, p.y_mm)
        if signed_dist <= -radius:
            factor = 1.0
        elif signed_dist >= radius:
            factor = 0.0
        else:
            factor = (radius - signed_dist) / (2.0 * radius)
        reaction = max(p.reaction_kN, 0.0)
        reduction_rows.append(
            {
                "pile": p.label,
                "factor": factor,
                "reaction_kN": reaction,
                "reduction_kN": factor * reaction,
                "distance_to_perimeter_mm": signed_dist,
            }
        )

    R_inside = sum(row["reduction_kN"] for row in reduction_rows)
    R_outside = sum(row["reaction_kN"] - row["reduction_kN"] for row in reduction_rows)

    Vu = max(Pu_total_kN - R_inside, 0.0)

    bo = 2.0 * (loaded_bx + d_avg_mm + loaded_by + d_avg_mm)

    return {
        "Vu_punch_kN": Vu,
        "R_inside_kN": R_inside,
        "R_outside_kN": R_outside,
        "bo_mm": bo,
        "x_crit_half_mm": x_lim,
        "y_crit_half_mm": y_lim,
        "piles_inside": ", ".join([row["pile"] for row in reduction_rows if row["factor"] >= 0.999]) or "-",
        "pile_reduction_factors": ", ".join([f"{row['pile']}={row['factor']:.2f}" for row in reduction_rows]),
    }


def stm_advisory(
    piles: List[PilePoint],
    geom: Geometry,
    mat: Material,
    d_avg_mm: float,
) -> pd.DataFrame:
    """
    Practical MacGregor-style STM advisory table.

    The load path is idealized from loaded area centroid to each pile head.
    For each pile:
        a = horizontal distance from column/pedestal centroid to pile center, not less than small number.
        theta = atan(d / a)
        T_est = R * cot(theta), an indicative tie force along bottom reinforcement direction.

    The vector is split into x/y components based on pile plan location.
    """
    rows = []
    for p in piles:
        dx = p.x_mm
        dy = p.y_mm
        a = math.sqrt(dx**2 + dy**2)
        if a < 1.0:
            a = 1.0
        theta_rad = math.atan2(d_avg_mm, a)
        theta_deg = math.degrees(theta_rad)
        cot_theta = 1.0 / max(math.tan(theta_rad), 1e-9)
        R = max(p.reaction_kN, 0.0)
        T = R * cot_theta
        ux = abs(dx) / max(a, 1e-9)
        uy = abs(dy) / max(a, 1e-9)
        Tx = T * ux
        Ty = T * uy

        if theta_deg < 25.0:
            note = "Shallow strut; increase depth or revise layout"
            stt = "WARN"
        elif theta_deg > 65.0:
            note = "Steep strut; check node geometry/detailing"
            stt = "WARN"
        else:
            note = "Practical STM angle range"
            stt = "OK"

        rows.append(
            {
                "Pile": p.label,
                "R (kN)": R,
                "a horizontal (mm)": a,
                "theta (deg)": theta_deg,
                "T tie estimate (kN)": T,
                "Tx component (kN)": Tx,
                "Ty component (kN)": Ty,
                "Status": stt,
                "Note": note,
            }
        )

    return pd.DataFrame(rows)


def design_all(state: DesignState, use_self_weight: bool = True) -> Dict[str, Any]:
    geom = state.geometry
    mat = state.material
    reinf = state.reinforcement

    cap_x = state.cap_length_x_mm
    cap_y = state.cap_width_y_mm
    d_x, d_y = effective_depths(geom, reinf)
    d_top_x, d_top_y = top_effective_depths(geom, reinf)
    d_avg = 0.5 * (d_x + d_y)

    Pu_total = state.loadcase.Pu_kN + (state.self_weight_kN if use_self_weight else 0.0)

    moment_demands = design_moments_from_pile_reactions(state.piles, geom, cap_x, cap_y)
    continuous_x = continuous_beam_strip_moments(state.piles, "x")
    continuous_y = continuous_beam_strip_moments(state.piles, "y")
    top_uplift_moments = top_moments_from_tension_pile_reactions(state.piles, geom)
    top_demand_x = governing_top_flexure_demand(continuous_x, top_uplift_moments["M_for_X_bars_kNm"])
    top_demand_y = governing_top_flexure_demand(continuous_y, top_uplift_moments["M_for_Y_bars_kNm"])
    shear_demands = one_way_shear_demands(state.piles, geom, d_x, d_y)
    punch_demand = punching_shear_demand(state.piles, geom, d_avg, Pu_total)

    # Flexure:
    # X bars use the cap width perpendicular to the X-bar span; Y bars use the
    # cap width perpendicular to the Y-bar span.
    flex_x = flexural_As_required(
        moment_demands["M_for_X_bars_kNm"], cap_y, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure
    )
    flex_y = flexural_As_required(
        moment_demands["M_for_Y_bars_kNm"], cap_x, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure
    )

    zero_top_flex = {"As_req_mm2": 0.0, "As_strength_mm2": 0.0, "As_min_mm2": 0.0, "rho": 0.0, "a_mm": 0.0, "phiMn_kNm": 0.0}
    top_flex_x = (
        flexural_As_required(top_demand_x["demand_kNm"], cap_y, d_top_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        if top_demand_x["demand_kNm"] > 1e-9 else dict(zero_top_flex)
    )
    top_flex_y = (
        flexural_As_required(top_demand_y["demand_kNm"], cap_x, d_top_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
        if top_demand_y["demand_kNm"] > 1e-9 else dict(zero_top_flex)
    )

    sp_x = provided_spacing_As(reinf.main_bar_x, reinf.spacing_x_mm, cap_y)
    sp_y = provided_spacing_As(reinf.main_bar_y, reinf.spacing_y_mm, cap_x)
    top_sp_x = provided_spacing_As(reinf.top_bar, reinf.top_spacing_mm, cap_y)
    top_sp_y = provided_spacing_As(reinf.top_bar, reinf.top_spacing_mm, cap_x)

    cap_xbars = flexural_capacity(sp_x["As_prov_mm2"], cap_y, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
    cap_ybars = flexural_capacity(sp_y["As_prov_mm2"], cap_x, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
    top_cap_xbars = flexural_capacity(top_sp_x["As_prov_mm2"], cap_y, d_top_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
    top_cap_ybars = flexural_capacity(top_sp_y["As_prov_mm2"], cap_x, d_top_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
    rho_x_provided = sp_x["As_prov_mm2"] / max(cap_y * d_x, 1e-9)
    rho_y_provided = sp_y["As_prov_mm2"] / max(cap_x * d_y, 1e-9)
    close_pile = close_pile_spacing_check(state.piles, geom)

    # One-way shear capacities by section orientation.
    ow_sec_normal_to_x = one_way_shear_capacity(
        cap_y, d_y, mat.fc_MPa, mat.lambda_c, mat.phi_shear, rho_w=rho_y_provided,
        use_close_pile_aci318_25=close_pile["applies"],
    )
    ow_sec_normal_to_y = one_way_shear_capacity(
        cap_x, d_x, mat.fc_MPa, mat.lambda_c, mat.phi_shear, rho_w=rho_x_provided,
        use_close_pile_aci318_25=close_pile["applies"],
    )

    # Punching capacity:
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm
    punch_cap = two_way_shear_capacity(
        punch_demand["bo_mm"],
        d_avg,
        mat.fc_MPa,
        mat.lambda_c,
        mat.phi_shear,
        loaded_bx_mm=loaded_bx,
        loaded_by_mm=loaded_by,
        column_location=geom.column_location,
        use_close_pile_aci318_25=close_pile["applies"],
    )

    # Bearing at column/pedestal:
    loaded_area = loaded_bx * loaded_by
    bearing = concrete_bearing_capacity(loaded_area, mat.fc_MPa, mat.phi_bearing)

    # Development lengths:
    db_x = bar_diameter(reinf.main_bar_x)
    db_y = bar_diameter(reinf.main_bar_y)
    ld_x = development_length_tension_estimate(
        db_x,
        mat.fc_MPa,
        mat.fy_MPa,
        mat.lambda_c,
        psi_t=1.0,
        psi_e=1.0,
        psi_s=0.8 if db_x <= 16.0 else 1.0,
        clear_cover_mm=min(geom.bottom_cover_mm, geom.side_cover_mm),
        clear_spacing_mm=max(sp_x["spacing_use_mm"] - db_x, 0.0),
    )
    ld_y = development_length_tension_estimate(
        db_y,
        mat.fc_MPa,
        mat.fy_MPa,
        mat.lambda_c,
        psi_t=1.0,
        psi_e=1.0,
        psi_s=0.8 if db_y <= 16.0 else 1.0,
        clear_cover_mm=min(geom.bottom_cover_mm, geom.side_cover_mm),
        clear_spacing_mm=max(sp_y["spacing_use_mm"] - db_y, 0.0),
    )
    db_top = bar_diameter(reinf.top_bar)
    ld_top = development_length_tension_estimate(
        db_top,
        mat.fc_MPa,
        mat.fy_MPa,
        mat.lambda_c,
        psi_t=1.0,
        psi_e=1.0,
        psi_s=0.8 if db_top <= 16.0 else 1.0,
        clear_cover_mm=min(geom.top_cover_mm, geom.side_cover_mm),
        clear_spacing_mm=max(reinf.top_spacing_mm - db_top, 0.0),
    )

    # STM advisory:
    stm = stm_advisory(state.piles, geom, mat, d_avg)
    T_x_total = stm["Tx component (kN)"].sum() if not stm.empty else 0.0
    T_y_total = stm["Ty component (kN)"].sum() if not stm.empty else 0.0
    # Advisory tie area uses one side of the additive component total to avoid
    # double-counting balanced left/right or top/bottom axial-load components.
    As_stm_for_x_bars = 0.5 * T_y_total * 1000.0 / max(mat.phi_stm_tie * mat.fy_MPa, 1e-9)
    As_stm_for_y_bars = 0.5 * T_x_total * 1000.0 / max(mat.phi_stm_tie * mat.fy_MPa, 1e-9)
    stm_statuses = set(stm["Status"].astype(str).str.upper().tolist()) if not stm.empty and "Status" in stm.columns else set()
    stm_angle_note = (
        "STM advisory angle range is OK."
        if stm_statuses == {"OK"}
        else "STM advisory has angle warnings; review the STM tab."
    )

    # Check results:
    checks = []
    compression_sum = sum(max(p.reaction_kN, 0.0) for p in state.piles)

    def sectional_shear_check(name: str, demand: float, capacity: float, note: str) -> CheckResult:
        if demand <= 1e-9 and compression_sum > 1e-9:
            return CheckResult(
                name,
                demand,
                capacity,
                np.nan,
                "kN",
                "STM",
                note + f" No pile reaction lies outside the d-from-face section; sectional shear D/C is not applicable. {stm_angle_note} Verify struts, ties, nodal zones, and anchorage.",
            )
        ratio = demand / max(capacity, 1e-9)
        return CheckResult(name, demand, capacity, ratio, "kN", status_from_ratio(ratio), note)

    def punching_check(name: str, demand: float, capacity: float, note: str) -> CheckResult:
        if demand <= 1e-9 and compression_sum > 1e-9 and punch_demand["R_inside_kN"] > 0.99 * compression_sum:
            return CheckResult(
                name,
                demand,
                capacity,
                np.nan,
                "kN",
                "STM",
                note + f" Pile reactions are inside the punching perimeter; sectional punching D/C is not applicable. {stm_angle_note} Verify struts, ties, nodal zones, and anchorage.",
            )
        ratio = demand / max(capacity, 1e-9)
        return CheckResult(name, demand, capacity, ratio, "kN", status_from_ratio(ratio), note)

    def top_flexure_check(name: str, demand_info: Dict[str, Any], capacity: float, direction_note: str) -> CheckResult:
        demand = float(demand_info.get("demand_kNm", 0.0))
        if demand <= 1e-9:
            return CheckResult(
                name,
                demand,
                capacity,
                np.nan,
                "kN-m",
                "NOMINAL",
                direction_note + " No governing uplift pile tension or interior pile-line negative moment in this direction; top bars are nominal/detailing.",
            )
        ratio = demand / max(capacity, 1e-9)
        source_note = (
            "Net uplift/tension pile demand controls."
            if demand_info.get("source") == "net_uplift_tension_pile"
            else "Continuous beam-strip negative moment controls."
        )
        return CheckResult(
            name,
            demand,
            capacity,
            ratio,
            "kN-m",
            status_from_ratio(ratio),
            f"{direction_note} {source_note}",
        )

    checks.append(
        CheckResult(
            "Pile compression",
            demand=max([p.reaction_kN for p in state.piles] + [0.0]),
            capacity=geom.pile_capacity_comp_kN,
            ratio=max([p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_comp_kN, 1e-9),
            unit="kN",
            status=status_from_ratio(max([p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_comp_kN, 1e-9)),
            note="Maximum compression pile reaction.",
        )
    )
    checks.append(
        CheckResult(
            "Pile tension/uplift",
            demand=max([-p.reaction_kN for p in state.piles] + [0.0]),
            capacity=geom.pile_capacity_tension_kN,
            ratio=max([-p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_tension_kN, 1e-9),
            unit="kN",
            status=status_from_ratio(max([-p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_tension_kN, 1e-9)),
            note="Maximum uplift reaction.",
        )
    )
    checks.append(
        CheckResult(
            "Flexure - bottom X bars",
            demand=moment_demands["M_for_X_bars_kNm"],
            capacity=cap_xbars["phiMn_kNm"],
            ratio=moment_demands["M_for_X_bars_kNm"] / max(cap_xbars["phiMn_kNm"], 1e-9),
            unit="kN-m",
            status=status_from_ratio(moment_demands["M_for_X_bars_kNm"] / max(cap_xbars["phiMn_kNm"], 1e-9)),
            note="Bottom bars parallel to X; cantilever action in Y.",
        )
    )
    checks.append(
        CheckResult(
            "Flexure - bottom Y bars",
            demand=moment_demands["M_for_Y_bars_kNm"],
            capacity=cap_ybars["phiMn_kNm"],
            ratio=moment_demands["M_for_Y_bars_kNm"] / max(cap_ybars["phiMn_kNm"], 1e-9),
            unit="kN-m",
            status=status_from_ratio(moment_demands["M_for_Y_bars_kNm"] / max(cap_ybars["phiMn_kNm"], 1e-9)),
            note="Bottom bars parallel to Y; cantilever action in X.",
        )
    )
    checks.append(
        top_flexure_check(
            "Flexure - top X bars",
            top_demand_x,
            top_cap_xbars["phiMn_kNm"],
            f"{top_demand_x['note']}",
        )
    )
    checks.append(
        top_flexure_check(
            "Flexure - top Y bars",
            top_demand_y,
            top_cap_ybars["phiMn_kNm"],
            f"{top_demand_y['note']}",
        )
    )
    checks.append(
        sectional_shear_check(
            "One-way shear - section normal to Y",
            shear_demands["V_for_X_bars_direction_kN"],
            ow_sec_normal_to_y["phiVc_kN"],
            "Critical section at d from loaded area face; full cap width used.",
        )
    )
    checks.append(
        sectional_shear_check(
            "One-way shear - section normal to X",
            shear_demands["V_for_Y_bars_direction_kN"],
            ow_sec_normal_to_x["phiVc_kN"],
            "Critical section at d from loaded area face; full cap width used.",
        )
    )
    checks.append(
        punching_check(
            "Two-way punching shear",
            punch_demand["Vu_punch_kN"],
            punch_cap["phiVc_kN"],
            (
                f"bo={punch_demand['bo_mm']:.0f} mm; vc governs by {punch_cap['governing_equation']}; "
                f"pile reduction factors: {punch_demand['pile_reduction_factors']}"
            ),
        )
    )
    checks.append(
        CheckResult(
            "Column/pedestal bearing",
            demand=Pu_total,
            capacity=bearing["phiPn_kN"],
            ratio=Pu_total / max(bearing["phiPn_kN"], 1e-9),
            unit="kN",
            status=status_from_ratio(Pu_total / max(bearing["phiPn_kN"], 1e-9)),
            note="Simplified concrete bearing under loaded area.",
        )
    )

    lateral_demand = max(abs(state.loadcase.Vux_kN), abs(state.loadcase.Vuy_kN), abs(state.loadcase.Tuz_kNm))
    if lateral_demand > 1e-9:
        checks.append(
            CheckResult(
                "Lateral/torsion design scope",
                demand=lateral_demand,
                capacity=0.0,
                ratio=np.inf,
                unit="kN or kN-m",
                status="FAIL",
                note="Vux, Vuy, and Tuz are imported/stored but require separate lateral pile group, pile head, and torsion design checks.",
            )
        )

    return {
        "cap_x_mm": cap_x,
        "cap_y_mm": cap_y,
        "d_x_mm": d_x,
        "d_y_mm": d_y,
        "d_top_x_mm": d_top_x,
        "d_top_y_mm": d_top_y,
        "d_avg_mm": d_avg,
        "Pu_total_kN": Pu_total,
        "moment_demands": moment_demands,
        "continuous_x": continuous_x,
        "continuous_y": continuous_y,
        "top_uplift_moments": top_uplift_moments,
        "top_demand_x": top_demand_x,
        "top_demand_y": top_demand_y,
        "close_pile_spacing": close_pile,
        "shear_demands": shear_demands,
        "punch_demand": punch_demand,
        "flex_x": flex_x,
        "flex_y": flex_y,
        "top_flex_x": top_flex_x,
        "top_flex_y": top_flex_y,
        "spacing_x": sp_x,
        "spacing_y": sp_y,
        "top_spacing_x": top_sp_x,
        "top_spacing_y": top_sp_y,
        "cap_xbars": cap_xbars,
        "cap_ybars": cap_ybars,
        "top_cap_xbars": top_cap_xbars,
        "top_cap_ybars": top_cap_ybars,
        "one_way_x": ow_sec_normal_to_y,
        "one_way_y": ow_sec_normal_to_x,
        "punch_cap": punch_cap,
        "bearing": bearing,
        "ld_x_mm": ld_x["ld_mm"],
        "ld_y_mm": ld_y["ld_mm"],
        "ld_top_mm": ld_top["ld_mm"],
        "ld_x": ld_x,
        "ld_y": ld_y,
        "ld_top": ld_top,
        "stm": stm,
        "As_stm_x_mm2": As_stm_for_x_bars,
        "As_stm_y_mm2": As_stm_for_y_bars,
        "As_stm_for_x_bars_mm2": As_stm_for_x_bars,
        "As_stm_for_y_bars_mm2": As_stm_for_y_bars,
        "checks": checks,
    }


# =============================================================================
# SAP2000 IMPORT
# =============================================================================

def read_uploaded_table(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = str(getattr(uploaded_file, "name", uploaded_file)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file, header=None)
        return normalize_sap2000_joint_reactions_table(raw)
    if name.endswith(".xlsx"):
        raw = pd.read_excel(uploaded_file, header=None)
        return normalize_sap2000_joint_reactions_table(raw)
    raise ValueError("Unsupported file type. Please use CSV or XLSX.")

def normalize_sap2000_joint_reactions_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SAP2000 exported TABLE: Joint Reactions sheets.

    Expected export shape:
        TABLE: Joint Reactions
        Joint | OutputCase | CaseType | F1 | F2 | F3 | M1 | M2 | M3
        Text  | Text       | Text     | KN | KN | KN | KN-m | KN-m | KN-m

    Also accepts already-clean tables with these columns.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    expected = ["Joint", "OutputCase", "CaseType", "F1", "F2", "F3", "M1", "M2", "M3"]
    existing_cols = [str(c).strip() for c in df.columns]
    if all(col in existing_cols for col in ["Joint", "OutputCase", "F3"]):
        out = df.copy()
        out.columns = existing_cols
    else:
        header_idx = None
        for idx, row in df.iterrows():
            vals = [str(v).strip() for v in row.tolist()]
            if "Joint" in vals and "OutputCase" in vals and "F3" in vals:
                header_idx = idx
                break
        if header_idx is None:
            # Fall back to first row as header for non-standard but table-like files.
            out = pd.read_excel(raw) if False else df
            return out

        header = [str(v).strip() if not pd.isna(v) else f"Column {i + 1}" for i, v in enumerate(df.loc[header_idx].tolist())]
        out = df.loc[header_idx + 1:].copy()
        out.columns = header

    out = out.dropna(axis=0, how="all")
    # Remove SAP units row and repeated title/header rows if pasted/exported together.
    first_col = out.columns[0]
    mask_units = out[first_col].astype(str).str.strip().str.lower().isin({"text", "joint"})
    out = out[~mask_units].copy()
    if "CaseType" in out.columns:
        out = out[out["CaseType"].astype(str).str.strip().str.lower() != "text"].copy()

    keep = [c for c in expected if c in out.columns]
    other = [c for c in out.columns if c not in keep]
    out = out[keep + other].reset_index(drop=True)

    for col in ["F1", "F2", "F3", "M1", "M2", "M3"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Joint" in out.columns:
        out["Joint"] = out["Joint"].astype(str).str.strip()
    if "OutputCase" in out.columns:
        out["OutputCase"] = out["OutputCase"].astype(str).str.strip()
    if "CaseType" in out.columns:
        out["CaseType"] = out["CaseType"].astype(str).str.strip()
    return out


def find_matching_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for a in aliases:
        key = a.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # More flexible contains search
    for c in columns:
        normalized = c.lower().replace(" ", "").replace("_", "")
        for a in aliases:
            aa = a.lower().replace(" ", "").replace("_", "")
            if normalized == aa or normalized.endswith(aa):
                return c
    return None


def sap2000_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "joint": find_matching_column(cols, ["Joint", "JointElm", "Point", "PointElm", "Unique Name"]),
        "case": find_matching_column(cols, ["OutputCase", "Load Case", "Case", "LoadCase", "Combo", "Combination"]),
        "case_type": find_matching_column(cols, ["CaseType", "Case Type", "OutputCaseType"]),
        "step": find_matching_column(cols, ["StepType", "Step", "Step Number", "StepNum"]),
        "F1": find_matching_column(cols, ["F1", "FX", "GlobalFX", "Reaction F1"]),
        "F2": find_matching_column(cols, ["F2", "FY", "GlobalFY", "Reaction F2"]),
        "F3": find_matching_column(cols, ["F3", "FZ", "GlobalFZ", "Reaction F3"]),
        "M1": find_matching_column(cols, ["M1", "MX", "GlobalMX", "Reaction M1"]),
        "M2": find_matching_column(cols, ["M2", "MY", "GlobalMY", "Reaction M2"]),
        "M3": find_matching_column(cols, ["M3", "MZ", "GlobalMZ", "Reaction M3"]),
    }


def build_loadcase_from_sap_row(
    row: pd.Series,
    cmap: Dict[str, Optional[str]],
    vertical_col: str,
    mx_col: str,
    my_col: str,
    vx_col: Optional[str],
    vy_col: Optional[str],
    torsion_col: Optional[str],
    vertical_sign: float,
    moment_sign: float,
    name: str,
) -> LoadCase:
    Pu = vertical_sign * safe_float(row.get(vertical_col, 0.0))
    Mux = moment_sign * safe_float(row.get(mx_col, 0.0))
    Muy = moment_sign * safe_float(row.get(my_col, 0.0))
    Vux = safe_float(row.get(vx_col, 0.0)) if vx_col else 0.0
    Vuy = safe_float(row.get(vy_col, 0.0)) if vy_col else 0.0
    Tuz = safe_float(row.get(torsion_col, 0.0)) if torsion_col else 0.0
    return LoadCase(name=name, Pu_kN=Pu, Mux_kNm=Mux, Muy_kNm=Muy, Vux_kN=Vux, Vuy_kN=Vuy, Tuz_kNm=Tuz)


def sap_envelope_table(df: pd.DataFrame, cmap: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    case_col = cmap.get("case")
    if case_col and case_col in df.columns:
        group_cols = [case_col]
    else:
        group_cols = []

    numeric_cols = [c for c in [cmap.get("F1"), cmap.get("F2"), cmap.get("F3"), cmap.get("M1"), cmap.get("M2"), cmap.get("M3")] if c]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if not numeric_cols:
        return pd.DataFrame()

    if group_cols:
        out_rows = []
        for key, g in df.groupby(group_cols):
            row = {"Case": key if isinstance(key, str) else key[0]}
            for c in numeric_cols:
                vals = pd.to_numeric(g[c], errors="coerce")
                if vals.dropna().empty:
                    continue
                idx_abs = vals.abs().idxmax()
                row[f"{c} abs max"] = vals.loc[idx_abs]
                row[f"{c} max"] = vals.max()
                row[f"{c} min"] = vals.min()
            out_rows.append(row)
        return pd.DataFrame(out_rows)
    else:
        row = {"Case": "All rows"}
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.dropna().empty:
                continue
            idx_abs = vals.abs().idxmax()
            row[f"{c} abs max"] = vals.loc[idx_abs]
            row[f"{c} max"] = vals.max()
            row[f"{c} min"] = vals.min()
        return pd.DataFrame([row])


def parse_joint_list(value: Any) -> List[str]:
    text = str(value or "").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def group_defaults_from_geometry(geom: Geometry, group_name: str = "Foundation 1", joint_ids: str = "") -> Dict[str, Any]:
    row = dict(GROUP_DEFAULT_COLUMNS)
    row.update(
        {
            "Group Name": group_name,
            "Joint IDs": joint_ids,
            "No. Piles": int(geom.n_piles),
            "Pile Dia (mm)": geom.pile_diameter_mm,
            "Spacing X (mm)": geom.spacing_x_mm,
            "Spacing Y (mm)": geom.spacing_y_mm,
            "Edge (mm)": geom.edge_from_pile_edge_mm,
            "Thickness (mm)": geom.cap_thickness_mm,
            "Pile Comp Cap (kN)": geom.pile_capacity_comp_kN,
            "Pile Tension Cap (kN)": geom.pile_capacity_tension_kN,
        }
    )
    return row


def ensure_group_columns(df: pd.DataFrame, geom: Geometry) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([group_defaults_from_geometry(geom)])
    out = df.copy()
    defaults = group_defaults_from_geometry(geom)
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    for col in ["Group Name", "Joint IDs"]:
        out[col] = out[col].fillna("").astype(str)
        out.loc[out[col].str.lower() == "nan", col] = ""
    numeric_cols = [col for col, default in defaults.items() if isinstance(default, (int, float)) and col in out.columns]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(defaults[col])
    return out[list(defaults.keys()) + [c for c in out.columns if c not in defaults]]


def apply_sidebar_geometry_to_groups(df: pd.DataFrame, geom: Geometry) -> pd.DataFrame:
    out = ensure_group_columns(df, geom).copy()
    sidebar_values = group_defaults_from_geometry(geom)
    for col in [
        "No. Piles",
        "Thickness (mm)",
        "Pile Dia (mm)",
        "Spacing X (mm)",
        "Spacing Y (mm)",
        "Edge (mm)",
        "Pile Comp Cap (kN)",
        "Pile Tension Cap (kN)",
    ]:
        out[col] = sidebar_values[col]
    return out


def group_geometry_override_warnings(df: pd.DataFrame, geom: Geometry) -> List[str]:
    if df is None or df.empty:
        return []
    checks = [
        ("No. Piles", geom.n_piles),
        ("Thickness (mm)", geom.cap_thickness_mm),
        ("Pile Dia (mm)", geom.pile_diameter_mm),
        ("Spacing X (mm)", geom.spacing_x_mm),
        ("Spacing Y (mm)", geom.spacing_y_mm),
        ("Edge (mm)", geom.edge_from_pile_edge_mm),
        ("Pile Comp Cap (kN)", geom.pile_capacity_comp_kN),
        ("Pile Tension Cap (kN)", geom.pile_capacity_tension_kN),
    ]
    notes = []
    for _, row in df.iterrows():
        name = str(row.get("Group Name", "Foundation")).strip() or "Foundation"
        diffs = []
        for col, sidebar_value in checks:
            if col in df.columns and abs(safe_float(row.get(col), sidebar_value) - float(sidebar_value)) > 1e-6:
                diffs.append(f"{col}: table {safe_float(row.get(col)):.0f}, sidebar {float(sidebar_value):.0f}")
        if diffs:
            notes.append(f"{name} is using table geometry overrides. Press APPLY to replace them with sidebar values. ({'; '.join(diffs)})")
    return notes


def geometry_from_group_row(base: Geometry, row: pd.Series) -> Geometry:
    return replace(
        base,
        n_piles=int(max(2, min(12, round(safe_float(row.get("No. Piles", base.n_piles), base.n_piles))))),
        pile_diameter_mm=safe_float(row.get("Pile Dia (mm)", base.pile_diameter_mm), base.pile_diameter_mm),
        spacing_x_mm=safe_float(row.get("Spacing X (mm)", base.spacing_x_mm), base.spacing_x_mm),
        spacing_y_mm=safe_float(row.get("Spacing Y (mm)", base.spacing_y_mm), base.spacing_y_mm),
        edge_from_pile_edge_mm=safe_float(row.get("Edge (mm)", base.edge_from_pile_edge_mm), base.edge_from_pile_edge_mm),
        cap_thickness_mm=safe_float(row.get("Thickness (mm)", base.cap_thickness_mm), base.cap_thickness_mm),
        pile_capacity_comp_kN=safe_float(row.get("Pile Comp Cap (kN)", base.pile_capacity_comp_kN), base.pile_capacity_comp_kN),
        pile_capacity_tension_kN=safe_float(row.get("Pile Tension Cap (kN)", base.pile_capacity_tension_kN), base.pile_capacity_tension_kN),
    )


def piles_from_group_geometry(geom: Geometry) -> List[PilePoint]:
    return make_piles(geom.n_piles, geom.spacing_x_mm, geom.spacing_y_mm, geom.pile_diameter_mm)


def default_sap_groups(df: pd.DataFrame, cmap: Dict[str, Optional[str]], geom: Geometry) -> pd.DataFrame:
    joint_col = cmap.get("joint")
    if df is None or df.empty or not joint_col or joint_col not in df.columns:
        return pd.DataFrame([group_defaults_from_geometry(geom, "Foundation 1", "")])
    joints = sorted({str(v).strip() for v in df[joint_col].dropna().tolist() if str(v).strip()})
    return pd.DataFrame([group_defaults_from_geometry(geom, "Foundation 1", ", ".join(joints[:12]))])


def default_manual_foundations(geom: Geometry) -> pd.DataFrame:
    row = group_defaults_from_geometry(geom, "Foundation 1", "")
    row.update(MANUAL_SERVICE_LOAD_DEFAULTS)
    return pd.DataFrame([row])


def normalize_manual_service_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for new_col, aliases in MANUAL_SERVICE_LOAD_ALIASES.items():
        if new_col not in work.columns:
            for old_col in aliases:
                if old_col in work.columns:
                    work[new_col] = work[old_col]
                    break
        if new_col not in work.columns:
            work[new_col] = MANUAL_SERVICE_LOAD_DEFAULTS.get(new_col, 0.0)
    for col, default in MANUAL_SERVICE_LOAD_DEFAULTS.items():
        if col not in work.columns:
            work[col] = default

    old_load_cols = {old_col for aliases in MANUAL_SERVICE_LOAD_ALIASES.values() for old_col in aliases}
    work = work.drop(columns=[col for col in old_load_cols if col in work.columns], errors="ignore")

    preferred_order = list(GROUP_DEFAULT_COLUMNS.keys()) + list(MANUAL_SERVICE_LOAD_DEFAULTS.keys())
    ordered_cols = [col for col in preferred_order if col in work.columns]
    ordered_cols += [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols]


def manual_service_ultimate_loadcases(row: pd.Series, group_name: str) -> Tuple[LoadCase, LoadCase]:
    d_ps = safe_float(row.get("D Ps (kN)", row.get("D Pu (kN)", row.get("Pu (kN)", 0.0))))
    d_msx = safe_float(row.get("D Msx (kN-m)", row.get("D Mux (kN-m)", row.get("Mux (kN-m)", 0.0))))
    d_msy = safe_float(row.get("D Msy (kN-m)", row.get("D Muy (kN-m)", row.get("Muy (kN-m)", 0.0))))
    l_ps = safe_float(row.get("L Ps (kN)", row.get("L Pu (kN)", 0.0)))
    l_msx = safe_float(row.get("L Msx (kN-m)", row.get("L Mux (kN-m)", 0.0)))
    l_msy = safe_float(row.get("L Msy (kN-m)", row.get("L Muy (kN-m)", 0.0)))
    fd = safe_float(row.get("D Factor", 1.2), 1.2)
    fl = safe_float(row.get("L Factor", 1.6), 1.6)
    service = LoadCase(
        name=f"{group_name} - Service D+L",
        Pu_kN=d_ps + l_ps,
        Mux_kNm=d_msx + l_msx,
        Muy_kNm=d_msy + l_msy,
    )
    ultimate = LoadCase(
        name=f"{group_name} - Ultimate {fd:g}D+{fl:g}L",
        Pu_kN=fd * d_ps + fl * l_ps,
        Mux_kNm=fd * d_msx + fl * l_msx,
        Muy_kNm=fd * d_msy + fl * l_msy,
    )
    return service, ultimate


def sap_loadcases_from_rows(
    df: pd.DataFrame,
    cmap: Dict[str, Optional[str]],
    vertical_col: str,
    mx_col: str,
    my_col: str,
    vx_col: Optional[str],
    vy_col: Optional[str],
    torsion_col: Optional[str],
    vertical_sign: float,
    moment_sign: float,
    group_name: str = "",
) -> List[LoadCase]:
    if df is None or df.empty:
        return []
    case_col = cmap.get("case")
    case_type_col = cmap.get("case_type")
    joint_col = cmap.get("joint")
    step_col = cmap.get("step")
    value_cols = [vertical_col, mx_col, my_col, vx_col, vy_col, torsion_col]
    value_cols = [c for c in value_cols if c and c in df.columns]
    work = df.copy()
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    # A selected group usually represents a foundation type, not one physical
    # cap combining every selected SAP joint. Keep each Joint+OutputCase as a
    # separate design case and envelope the results. Sum only duplicate rows
    # for the same joint and output case, for example step/table duplicates.
    group_cols = []
    if joint_col and joint_col in work.columns:
        group_cols.append(joint_col)
    if case_col and case_col in work.columns:
        group_cols.append(case_col)
    if case_type_col and case_type_col in work.columns:
        group_cols.append(case_type_col)
    if step_col and step_col in work.columns:
        group_cols.append(step_col)
    grouped = work.groupby(group_cols, dropna=False)[value_cols].sum().reset_index() if group_cols else work

    out = []
    for idx, row in grouped.iterrows():
        parts = []
        if joint_col and joint_col in grouped.columns:
            parts.append(f"J{str(row.get(joint_col)).strip()}")
        if case_col and case_col in grouped.columns:
            parts.append(str(row.get(case_col)).strip())
        if step_col and step_col in grouped.columns:
            step_value = str(row.get(step_col)).strip()
            if step_value and step_value.lower() != "nan":
                parts.append(step_value)
        if not parts:
            parts.append(f"SAP row {idx}")
        case_name = " - ".join(parts)
        out.append(
            build_loadcase_from_sap_row(
                row=row,
                cmap=cmap,
                vertical_col=vertical_col,
                mx_col=mx_col,
                my_col=my_col,
                vx_col=vx_col,
                vy_col=vy_col,
                torsion_col=torsion_col,
                vertical_sign=vertical_sign,
                moment_sign=moment_sign,
                name=case_name,
            )
        )
    return out


def prepare_design_state(
    mat: Material,
    geom: Geometry,
    reinf: Reinforcement,
    piles_base: List[PilePoint],
    loadcase: LoadCase,
    include_self_weight: bool,
) -> Tuple[DesignState, Dict[str, Any]]:
    cap_x, cap_y = cap_dimensions_from_piles(piles_base, geom.edge_from_pile_edge_mm)
    sw = self_weight_cap_kN(cap_x, cap_y, geom.cap_thickness_mm, mat.gamma_conc_kN_m3)
    self_weight = sw if include_self_weight else 0.0
    total_Pu = loadcase.Pu_kN + self_weight
    piles_with_reactions = distribute_vertical_load_to_piles(
        piles_base,
        total_Pu,
        loadcase.Mux_kNm,
        loadcase.Muy_kNm,
    )
    state = DesignState(
        material=mat,
        geometry=geom,
        reinforcement=reinf,
        loadcase=loadcase,
        piles=piles_with_reactions,
        cap_length_x_mm=cap_x,
        cap_width_y_mm=cap_y,
        effective_depth_x_mm=effective_depths(geom, reinf)[0],
        effective_depth_y_mm=effective_depths(geom, reinf)[1],
        self_weight_kN=self_weight,
    )
    return state, design_all(state, use_self_weight=include_self_weight)


def max_check_ratio(results: Dict[str, Any]) -> float:
    return max([c.ratio for c in results.get("checks", []) if np.isfinite(c.ratio)] + [0.0])


def design_input_signature(
    mat: Material,
    geom: Geometry,
    reinf: Reinforcement,
    include_self_weight: bool,
    load_input: Dict[str, Any],
) -> str:
    groups = load_input.get("groups", pd.DataFrame())
    groups_json = groups.to_json(orient="split", default_handler=str) if isinstance(groups, pd.DataFrame) else ""
    payload = {
        "material": asdict(mat),
        "geometry": asdict(geom),
        "reinforcement": asdict(reinf),
        "include_self_weight": include_self_weight,
        "mode": load_input.get("mode"),
        "groups": groups_json,
        "sap_rows": len(load_input.get("sap_df", pd.DataFrame())) if isinstance(load_input.get("sap_df", None), pd.DataFrame) else 0,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def pile_spacing_warnings(geom: Geometry) -> List[str]:
    notes = []
    if geom.spacing_x_mm < 3.0 * geom.pile_diameter_mm:
        notes.append("Spacing X is less than 3D; check pile clear spacing and group effects.")
    if geom.spacing_y_mm < 3.0 * geom.pile_diameter_mm:
        notes.append("Spacing Y is less than 3D; check pile clear spacing and group effects.")
    if geom.edge_from_pile_edge_mm < 0.5 * geom.pile_diameter_mm:
        notes.append("Edge distance is less than 0.5D from pile edge; verify detailing and concrete breakout.")
    return notes


def is_service_loadcase(loadcase: LoadCase) -> bool:
    return "CS-" in str(loadcase.name).upper()


def is_ultimate_loadcase(loadcase: LoadCase) -> bool:
    return "CU-" in str(loadcase.name).upper()


def envelope_group_design(
    group_name: str,
    loadcases: List[LoadCase],
    mat: Material,
    geom: Geometry,
    reinf: Reinforcement,
    piles_base: List[PilePoint],
    include_self_weight: bool,
    service_loadcases: Optional[List[LoadCase]] = None,
    ultimate_loadcases: Optional[List[LoadCase]] = None,
) -> Dict[str, Any]:
    service_cases = service_loadcases if service_loadcases is not None else [lc for lc in loadcases if is_service_loadcase(lc)]
    ultimate_cases = ultimate_loadcases if ultimate_loadcases is not None else [lc for lc in loadcases if is_ultimate_loadcase(lc)]
    if service_loadcases is None and not service_cases:
        service_cases = loadcases
    if not ultimate_cases:
        ultimate_cases = loadcases

    service_runs = []
    for lc in service_cases:
        state, results = prepare_design_state(mat, geom, reinf, piles_base, lc, include_self_weight)
        service_runs.append({"group": group_name, "loadcase": lc.name, "state": state, "results": results})

    rc_runs = []
    for lc in ultimate_cases:
        state, results = prepare_design_state(mat, geom, reinf, piles_base, lc, include_self_weight)
        rc_runs.append({"group": group_name, "loadcase": lc.name, "state": state, "results": results})

    if not service_runs:
        state, results = prepare_design_state(mat, geom, reinf, piles_base, LoadCase(name=f"{group_name} - default"), include_self_weight)
        service_runs.append({"group": group_name, "loadcase": state.loadcase.name, "state": state, "results": results})
    if not rc_runs:
        rc_runs = service_runs

    def governing(metric):
        return max(rc_runs, key=lambda r: metric(r["results"], r["state"]))

    overall = governing(lambda res, _state: max_check_ratio(res))
    flex_x_gov = governing(lambda res, _state: res["flex_x"]["As_req_mm2"])
    flex_y_gov = governing(lambda res, _state: res["flex_y"]["As_req_mm2"])
    top_x_gov = governing(lambda res, _state: res["top_demand_x"]["demand_kNm"] / max(res["top_cap_xbars"]["phiMn_kNm"], 1e-9))
    top_y_gov = governing(lambda res, _state: res["top_demand_y"]["demand_kNm"] / max(res["top_cap_ybars"]["phiMn_kNm"], 1e-9))
    punch_gov = governing(lambda res, _state: res["punch_demand"]["Vu_punch_kN"] / max(res["punch_cap"]["phiVc_kN"], 1e-9))
    shear_x_gov = governing(lambda res, _state: res["shear_demands"]["V_for_X_bars_direction_kN"] / max(res["one_way_x"]["phiVc_kN"], 1e-9))
    shear_y_gov = governing(lambda res, _state: res["shear_demands"]["V_for_Y_bars_direction_kN"] / max(res["one_way_y"]["phiVc_kN"], 1e-9))

    pile_rows = []
    labels = [p.label for p in piles_base]
    for label in labels:
        vals = []
        coords = None
        dia = geom.pile_diameter_mm
        for run in service_runs:
            pile = next((p for p in run["state"].piles if p.label == label), None)
            if pile is None:
                continue
            vals.append((pile.reaction_kN, run["loadcase"]))
            coords = (pile.x_mm, pile.y_mm)
            dia = pile.diameter_mm
        if not vals:
            continue
        comp = max(vals, key=lambda item: item[0])
        ten = min(vals, key=lambda item: item[0])
        pile_rows.append(
            {
                "Group": group_name,
                "Pile": label,
                "x (mm)": coords[0] if coords else 0.0,
                "y (mm)": coords[1] if coords else 0.0,
                "Max compression (kN)": max(comp[0], 0.0),
                "Compression combo": comp[1],
                "Max uplift (kN)": max(-ten[0], 0.0),
                "Uplift combo": ten[1],
                "Compression ratio": max(comp[0], 0.0) / max(geom.pile_capacity_comp_kN, 1e-9),
                "Tension ratio": max(-ten[0], 0.0) / max(geom.pile_capacity_tension_kN, 1e-9),
                "diameter_mm": dia,
            }
        )

    rc_pile_rows = []
    for label in labels:
        vals = []
        coords = None
        dia = geom.pile_diameter_mm
        for run in rc_runs:
            pile = next((p for p in run["state"].piles if p.label == label), None)
            if pile is None:
                continue
            vals.append((pile.reaction_kN, run["loadcase"]))
            coords = (pile.x_mm, pile.y_mm)
            dia = pile.diameter_mm
        if not vals:
            continue
        comp = max(vals, key=lambda item: item[0])
        ten = min(vals, key=lambda item: item[0])
        rc_pile_rows.append(
            {
                "Group": group_name,
                "Pile": label,
                "x (mm)": coords[0] if coords else 0.0,
                "y (mm)": coords[1] if coords else 0.0,
                "Max compression (kN)": max(comp[0], 0.0),
                "Compression combo": comp[1],
                "Max uplift (kN)": max(-ten[0], 0.0),
                "Uplift combo": ten[1],
                "diameter_mm": dia,
            }
        )

    def pile_reactions_by_combo_rows(runs: List[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
        rows = []
        for run in runs:
            for pile in run["state"].piles:
                rows.append(
                    {
                        "Group": group_name,
                        "Design phase": phase,
                        "Combo": run["loadcase"],
                        "Pile": pile.label,
                        "x (mm)": pile.x_mm,
                        "y (mm)": pile.y_mm,
                        "R, compression + (kN)": pile.reaction_kN,
                        "Compression demand (kN)": max(pile.reaction_kN, 0.0),
                        "Uplift demand (kN)": max(-pile.reaction_kN, 0.0),
                    }
                )
        return rows

    service_reaction_rows = pile_reactions_by_combo_rows(service_runs, "Pile capacity")
    rc_reaction_rows = pile_reactions_by_combo_rows(rc_runs, "RC design")

    display_state = overall["state"]
    display_state = DesignState(
        material=display_state.material,
        geometry=display_state.geometry,
        reinforcement=display_state.reinforcement,
        loadcase=LoadCase(name=f"{group_name} envelope", Pu_kN=display_state.loadcase.Pu_kN, Mux_kNm=display_state.loadcase.Mux_kNm, Muy_kNm=display_state.loadcase.Muy_kNm),
        piles=display_state.piles,
        cap_length_x_mm=display_state.cap_length_x_mm,
        cap_width_y_mm=display_state.cap_width_y_mm,
        effective_depth_x_mm=display_state.effective_depth_x_mm,
        effective_depth_y_mm=display_state.effective_depth_y_mm,
        self_weight_kN=display_state.self_weight_kN,
    )

    summary = {
        "Group": group_name,
        "No. Piles": geom.n_piles,
        "Pile Dia (mm)": geom.pile_diameter_mm,
        "Cap X (mm)": display_state.cap_length_x_mm,
        "Cap Y (mm)": display_state.cap_width_y_mm,
        "Thickness (mm)": geom.cap_thickness_mm,
        "Geometry warnings": "; ".join(pile_spacing_warnings(geom)),
        "Service cases checked": len(service_runs),
        "Ultimate cases checked": len(rc_runs),
        "RC governing combo": overall["loadcase"],
        "Max D/C": max_check_ratio(overall["results"]),
        "X bars governing combo": flex_x_gov["loadcase"],
        "X bars As req (mm2)": flex_x_gov["results"]["flex_x"]["As_req_mm2"],
        "Y bars governing combo": flex_y_gov["loadcase"],
        "Y bars As req (mm2)": flex_y_gov["results"]["flex_y"]["As_req_mm2"],
        "Top X governing combo": top_x_gov["loadcase"],
        "Top X M- (kN-m)": top_x_gov["results"]["top_demand_x"]["demand_kNm"],
        "Top X source": top_x_gov["results"]["top_demand_x"]["source"],
        "Top X D/C": top_x_gov["results"]["top_demand_x"]["demand_kNm"] / max(top_x_gov["results"]["top_cap_xbars"]["phiMn_kNm"], 1e-9) if top_x_gov["results"]["top_demand_x"]["demand_kNm"] > 1e-9 else np.nan,
        "Top Y governing combo": top_y_gov["loadcase"],
        "Top Y M- (kN-m)": top_y_gov["results"]["top_demand_y"]["demand_kNm"],
        "Top Y source": top_y_gov["results"]["top_demand_y"]["source"],
        "Top Y D/C": top_y_gov["results"]["top_demand_y"]["demand_kNm"] / max(top_y_gov["results"]["top_cap_ybars"]["phiMn_kNm"], 1e-9) if top_y_gov["results"]["top_demand_y"]["demand_kNm"] > 1e-9 else np.nan,
        "Punching governing combo": punch_gov["loadcase"],
        "Punching D/C": punch_gov["results"]["punch_demand"]["Vu_punch_kN"] / max(punch_gov["results"]["punch_cap"]["phiVc_kN"], 1e-9),
        "One-way X governing combo": shear_x_gov["loadcase"],
        "One-way Y governing combo": shear_y_gov["loadcase"],
    }
    display_results = dict(overall["results"])
    for key in ["flex_x", "spacing_x", "cap_xbars"]:
        display_results[key] = flex_x_gov["results"][key]
    for key in ["flex_y", "spacing_y", "cap_ybars"]:
        display_results[key] = flex_y_gov["results"][key]
    for key in ["top_flex_x", "top_spacing_x", "top_cap_xbars", "d_top_x_mm", "ld_top_mm", "ld_top", "continuous_x", "top_uplift_moments", "top_demand_x"]:
        display_results[key] = top_x_gov["results"][key]
    for key in ["top_flex_y", "top_spacing_y", "top_cap_ybars", "d_top_y_mm", "continuous_y", "top_demand_y"]:
        display_results[key] = top_y_gov["results"][key]
    for key in ["punch_demand", "punch_cap"]:
        display_results[key] = punch_gov["results"][key]
    display_results["one_way_x"] = shear_x_gov["results"]["one_way_x"]
    display_results["one_way_y"] = shear_y_gov["results"]["one_way_y"]
    display_results["shear_demands"] = dict(overall["results"]["shear_demands"])
    display_results["shear_demands"]["V_for_X_bars_direction_kN"] = shear_x_gov["results"]["shear_demands"]["V_for_X_bars_direction_kN"]
    display_results["shear_demands"]["V_for_Y_bars_direction_kN"] = shear_y_gov["results"]["shear_demands"]["V_for_Y_bars_direction_kN"]
    display_results["governing_combos"] = {
        "pile_service": max(pile_rows, key=lambda row: row["Max compression (kN)"])["Compression combo"] if pile_rows else "-",
        "rc_overall": overall["loadcase"],
        "flex_x_ultimate": flex_x_gov["loadcase"],
        "flex_y_ultimate": flex_y_gov["loadcase"],
        "top_flex_x_ultimate": top_x_gov["loadcase"],
        "top_flex_y_ultimate": top_y_gov["loadcase"],
        "punching_ultimate": punch_gov["loadcase"],
        "one_way_x_ultimate": shear_x_gov["loadcase"],
        "one_way_y_ultimate": shear_y_gov["loadcase"],
        "pile_compression_envelope": max(pile_rows, key=lambda row: row["Max compression (kN)"])["Compression combo"] if pile_rows else "-",
        "pile_uplift_envelope": max(pile_rows, key=lambda row: row["Max uplift (kN)"])["Uplift combo"] if pile_rows else "-",
        "rc_pile_compression_envelope": max(rc_pile_rows, key=lambda row: row["Max compression (kN)"])["Compression combo"] if rc_pile_rows else "-",
        "rc_pile_uplift_envelope": max(rc_pile_rows, key=lambda row: row["Max uplift (kN)"])["Uplift combo"] if rc_pile_rows else "-",
    }
    max_comp = max([row["Max compression (kN)"] for row in pile_rows] + [0.0])
    max_uplift = max([row["Max uplift (kN)"] for row in pile_rows] + [0.0])
    envelope_checks = []
    for check in overall["results"]["checks"]:
        if check.name == "Pile compression":
            ratio = max_comp / max(geom.pile_capacity_comp_kN, 1e-9)
            envelope_checks.append(CheckResult(check.name, max_comp, geom.pile_capacity_comp_kN, ratio, "kN", status_from_ratio(ratio), "Envelope maximum compression pile reaction."))
        elif check.name == "Pile tension/uplift":
            ratio = max_uplift / max(geom.pile_capacity_tension_kN, 1e-9)
            envelope_checks.append(CheckResult(check.name, max_uplift, geom.pile_capacity_tension_kN, ratio, "kN", status_from_ratio(ratio), "Envelope maximum uplift pile reaction."))
        else:
            envelope_checks.append(check)
    pile_envelope_df = pd.DataFrame(pile_rows)
    rc_pile_envelope_df = pd.DataFrame(rc_pile_rows)
    display_results["checks"] = envelope_checks
    display_results["pile_reaction_envelope"] = pile_envelope_df
    display_results["rc_pile_reaction_envelope"] = rc_pile_envelope_df
    display_results["pile_reactions_by_combo"] = pd.DataFrame(service_reaction_rows)
    display_results["rc_pile_reactions_by_combo"] = pd.DataFrame(rc_reaction_rows)
    return {
        "group": group_name,
        "runs": rc_runs,
        "service_runs": service_runs,
        "ultimate_runs": rc_runs,
        "state": display_state,
        "results": display_results,
        "summary": summary,
        "pile_envelope": pile_envelope_df,
        "rc_pile_envelope": rc_pile_envelope_df,
    }


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def add_dim_line(ax, p1, p2, text, offset=(0, 0), text_offset=(0, 0), fontsize=8):
    x1, y1 = p1
    x2, y2 = p2
    ox, oy = offset
    tx, ty = text_offset
    ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color="black", lw=0.8)
    ax.add_patch(FancyArrowPatch((x1 + ox, y1 + oy), (x1, y1), arrowstyle="-", mutation_scale=6, lw=0.6))
    ax.add_patch(FancyArrowPatch((x2 + ox, y2 + oy), (x2, y2), arrowstyle="-", mutation_scale=6, lw=0.6))
    ax.text((x1 + x2) / 2 + ox + tx, (y1 + y2) / 2 + oy + ty, text, ha="center", va="center", fontsize=fontsize)


def plot_plan(state: DesignState, results: Dict[str, Any], show_rebar: bool = True, show_reactions: bool = True):
    geom = state.geometry
    reinf = state.reinforcement
    cap_x = state.cap_length_x_mm
    cap_y = state.cap_width_y_mm

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_aspect("equal", adjustable="box")

    # Cap boundary
    ax.add_patch(Rectangle((-cap_x / 2, -cap_y / 2), cap_x, cap_y, fill=False, lw=2.0))
    ax.text(-cap_x / 2, cap_y / 2 + 170, f"PILE CAP {cap_x:.0f} x {cap_y:.0f} x {geom.cap_thickness_mm:.0f} mm", fontsize=10, weight="bold")

    # Column/pedestal
    ax.add_patch(Rectangle((-geom.column_bx_mm / 2, -geom.column_by_mm / 2), geom.column_bx_mm, geom.column_by_mm, fill=False, lw=1.8, linestyle="-"))
    ax.text(0, 0, "COLUMN", ha="center", va="center", fontsize=8)

    if geom.use_pedestal_for_shear and (geom.pedestal_bx_mm != geom.column_bx_mm or geom.pedestal_by_mm != geom.column_by_mm):
        ax.add_patch(Rectangle((-geom.pedestal_bx_mm / 2, -geom.pedestal_by_mm / 2), geom.pedestal_bx_mm, geom.pedestal_by_mm, fill=False, lw=1.2, linestyle="--"))
        ax.text(0, -geom.pedestal_by_mm / 2 - 100, "PEDESTAL / LOADED AREA", ha="center", va="top", fontsize=7)

    # Critical punching perimeter
    pdm = results.get("punch_demand", {})
    if pdm:
        xlim = pdm["x_crit_half_mm"]
        ylim = pdm["y_crit_half_mm"]
        ax.add_patch(Rectangle((-xlim, -ylim), 2 * xlim, 2 * ylim, fill=False, lw=1.0, linestyle=":", alpha=0.8))
        ax.text(xlim + 50, ylim + 50, "d/2 punching perimeter", fontsize=7, va="bottom")

    # Piles
    for p in state.piles:
        ax.add_patch(Circle((p.x_mm, p.y_mm), p.diameter_mm / 2, fill=False, lw=1.6))
        ax.plot(p.x_mm, p.y_mm, marker="x", ms=5)
        label = f"{p.label}"
        if show_reactions:
            label += f"\nR={p.reaction_kN:.0f} kN"
        ax.text(p.x_mm, p.y_mm, label, ha="center", va="center", fontsize=7)

    # Rebar schematic
    if show_rebar:
        cover = geom.side_cover_mm
        sx = results["spacing_x"]["spacing_use_mm"]
        sy = results["spacing_y"]["spacing_use_mm"]

        # X bars: horizontal lines across cap, spaced along y
        y_values = np.arange(-cap_y / 2 + cover, cap_y / 2 - cover + 1, sx)
        for y in y_values:
            ax.plot([-cap_x / 2 + cover, cap_x / 2 - cover], [y, y], lw=0.35, alpha=0.5)
        # Y bars: vertical lines across cap, spaced along x
        x_values = np.arange(-cap_x / 2 + cover, cap_x / 2 - cover + 1, sy)
        for x in x_values:
            ax.plot([x, x], [-cap_y / 2 + cover, cap_y / 2 - cover], lw=0.35, alpha=0.5)

        ax.text(
            -cap_x / 2 + 80,
            -cap_y / 2 + 80,
            f"BOT. X: {reinf.main_bar_x}@{sx:.0f}\nBOT. Y: {reinf.main_bar_y}@{sy:.0f}",
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.75),
        )

    # Axes
    axis_len = min(cap_x, cap_y) * 0.18
    ax.arrow(0, 0, axis_len, 0, head_width=60, head_length=80, length_includes_head=True)
    ax.arrow(0, 0, 0, axis_len, head_width=60, head_length=80, length_includes_head=True)
    ax.text(axis_len + 80, 0, "X", va="center")
    ax.text(0, axis_len + 80, "Y", ha="center")

    # Dimensions
    dim_offset_y = -cap_y / 2 - 350
    add_dim_line(ax, (-cap_x / 2, -cap_y / 2), (cap_x / 2, -cap_y / 2), f"{cap_x:.0f}", offset=(0, -250), text_offset=(0, -35))
    add_dim_line(ax, (cap_x / 2, -cap_y / 2), (cap_x / 2, cap_y / 2), f"{cap_y:.0f}", offset=(250, 0), text_offset=(55, 0))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.25, lw=0.5)
    margin = max(800, 0.12 * max(cap_x, cap_y))
    ax.set_xlim(-cap_x / 2 - margin, cap_x / 2 + margin)
    ax.set_ylim(-cap_y / 2 - margin, cap_y / 2 + margin)
    ax.set_title("Drawing-style Plan View", fontsize=12, weight="bold")
    return fig


def plot_elevation(state: DesignState, results: Dict[str, Any], direction: str = "X"):
    geom = state.geometry
    reinf = state.reinforcement
    cap_x = state.cap_length_x_mm if direction.upper() == "X" else state.cap_width_y_mm
    cap_y = state.cap_width_y_mm if direction.upper() == "X" else state.cap_length_x_mm
    thickness = geom.cap_thickness_mm

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(Rectangle((-cap_x / 2, 0), cap_x, thickness, fill=False, lw=2.0))
    ax.text(-cap_x / 2, thickness + 120, f"ELEVATION {direction.upper()}-DIRECTION", fontsize=10, weight="bold")

    # Column projection
    col_width = geom.column_bx_mm if direction.upper() == "X" else geom.column_by_mm
    ax.add_patch(Rectangle((-col_width / 2, thickness), col_width, thickness * 0.35, fill=False, lw=1.5))
    ax.text(0, thickness * 1.18, "COLUMN", ha="center", va="center", fontsize=8)

    # Piles along selected axis shown if near centerline in perpendicular axis.
    if direction.upper() == "X":
        axis_coord = [(p.x_mm, p.y_mm, p) for p in state.piles]
        perp_tol = max(geom.spacing_y_mm * 0.55, geom.pile_diameter_mm)
    else:
        axis_coord = [(p.y_mm, p.x_mm, p) for p in state.piles]
        perp_tol = max(geom.spacing_x_mm * 0.55, geom.pile_diameter_mm)

    for x, perp, p in axis_coord:
        if abs(perp) <= perp_tol:
            ax.add_patch(Rectangle((x - p.diameter_mm / 2, -thickness * 0.50), p.diameter_mm, thickness * 0.50, fill=False, lw=1.2))
            ax.text(x, -thickness * 0.25, p.label, ha="center", va="center", fontsize=7)

    # Bottom and top bars schematic
    cover_bot = geom.bottom_cover_mm
    cover_top = geom.top_cover_mm
    ax.plot([-cap_x / 2 + geom.side_cover_mm, cap_x / 2 - geom.side_cover_mm], [cover_bot, cover_bot], lw=2.0)
    ax.plot([-cap_x / 2 + geom.side_cover_mm, cap_x / 2 - geom.side_cover_mm], [thickness - cover_top, thickness - cover_top], lw=1.0, linestyle="--")

    bot_label = f"BOT. MAIN {reinf.main_bar_x if direction.upper() == 'X' else reinf.main_bar_y}"
    ax.text(-cap_x / 2 + 80, cover_bot + 60, bot_label, fontsize=8)
    ax.text(-cap_x / 2 + 80, thickness - cover_top - 100, f"TOP BARS {reinf.top_bar}@{reinf.top_spacing_mm:.0f}", fontsize=8)

    # Depth dimension
    add_dim_line(ax, (cap_x / 2, 0), (cap_x / 2, thickness), f"h = {thickness:.0f}", offset=(250, 0), text_offset=(70, 0))
    add_dim_line(ax, (-cap_x / 2, 0), (cap_x / 2, 0), f"{cap_x:.0f}", offset=(0, -220), text_offset=(0, -40))

    ax.set_xlim(-cap_x / 2 - 700, cap_x / 2 + 850)
    ax.set_ylim(-thickness * 0.65, thickness * 1.55)
    ax.axis("off")
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def checks_to_dataframe(checks: List[CheckResult]) -> pd.DataFrame:
    rows = []
    for c in checks:
        rows.append(
            {
                "Check": c.name,
                "Demand": c.demand,
                "Capacity": c.capacity,
                "Ratio": c.ratio,
                "Unit": c.unit,
                "Status": c.status,
                "Note": c.note,
            }
        )
    return pd.DataFrame(rows)



def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"

    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: fmt(value, 3) if isinstance(value, float) else str(value))

    columns = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_markdown_report(state: DesignState, results: Dict[str, Any]) -> str:
    checks_df = checks_to_dataframe(results["checks"])
    pile_df = pile_group_result_table(state.piles, state.geometry)
    stm_df = results["stm"]

    lines = []
    lines.append(f"# {APP_TITLE} Report")
    lines.append("")
    lines.append("## Design Basis")
    lines.append(f"- Design approach: ACI 318-25 style strength design plus practical MacGregor-style pile-cap load-path checks.")
    lines.append("- This report is a calculation aid and must be verified by the responsible engineer.")
    lines.append("")
    lines.append("## Materials")
    lines.append(f"- fc' = {state.material.fc_MPa:.1f} MPa")
    lines.append(f"- fy = {state.material.fy_MPa:.1f} MPa")
    lines.append(f"- lambda = {state.material.lambda_c:.2f}")
    lines.append(f"- phi flexure = {state.material.phi_flexure:.2f}")
    lines.append(f"- phi shear = {state.material.phi_shear:.2f}")
    lines.append("")
    lines.append("## Geometry")
    lines.append(f"- Number of piles = {state.geometry.n_piles}")
    lines.append(f"- Pile diameter = {state.geometry.pile_diameter_mm:.0f} mm")
    lines.append(f"- Cap size = {state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f} mm")
    lines.append(f"- Column = {state.geometry.column_bx_mm:.0f} x {state.geometry.column_by_mm:.0f} mm")
    lines.append(f"- Effective depth X bars = {results['d_x_mm']:.0f} mm")
    lines.append(f"- Effective depth Y bars = {results['d_y_mm']:.0f} mm")
    lines.append("")
    lines.append("## Loads")
    lc = state.loadcase
    lines.append(f"- Load case = {lc.name}")
    lines.append(f"- Pu = {lc.Pu_kN:.2f} kN")
    lines.append(f"- Mux = {lc.Mux_kNm:.2f} kN-m")
    lines.append(f"- Muy = {lc.Muy_kNm:.2f} kN-m")
    lines.append(f"- Self weight included in distribution = {state.self_weight_kN:.2f} kN")
    lines.append(f"- Total Pu used in checks = {results['Pu_total_kN']:.2f} kN")
    lines.append("")
    lines.append("## Main Reinforcement")
    lines.append(f"- Bottom X bars: {state.reinforcement.main_bar_x} @ {results['spacing_x']['spacing_use_mm']:.0f} mm")
    lines.append(f"- Bottom Y bars: {state.reinforcement.main_bar_y} @ {results['spacing_y']['spacing_use_mm']:.0f} mm")
    lines.append(f"- Top X/Y bars: {state.reinforcement.top_bar} @ {results['top_spacing_x']['spacing_use_mm']:.0f} mm")
    lines.append(f"- Development estimate X bars = {results['ld_x_mm']:.0f} mm (cb/db factor {results['ld_x']['confinement_factor']:.2f})")
    lines.append(f"- Development estimate Y bars = {results['ld_y_mm']:.0f} mm (cb/db factor {results['ld_y']['confinement_factor']:.2f})")
    lines.append(f"- Development estimate top bars = {results['ld_top_mm']:.0f} mm (cb/db factor {results['ld_top']['confinement_factor']:.2f})")
    lines.append("")
    lines.append("## Check Summary")
    lines.append(dataframe_to_markdown_table(checks_df))
    lines.append("")
    lines.append("## Pile Reaction Table")
    lines.append(dataframe_to_markdown_table(pile_df))
    lines.append("")
    lines.append("## STM Advisory")
    lines.append(dataframe_to_markdown_table(stm_df))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Verify ACI 318-25 clauses directly for final design.")
    lines.append("- Shear capacities use the ACI 318-25 close-pile provision when the pile spacing check is satisfied; otherwise they include lambda_s size effect.")
    lines.append("- Pile reactions near the punching perimeter are linearly reduced over one pile radius each side of the critical section.")
    lines.append("- For pile caps with small shear span-to-depth ratio, use a full strut-and-tie model.")
    lines.append("- For seismic regions, pile anchorage, confinement, tie forces, shear friction, and ductile detailing may govern.")
    lines.append("- Check geotechnical pile compression, uplift, lateral load, settlement, group effects, and construction tolerances.")
    return "\n".join(lines)


def pdf_safe(value: Any) -> str:
    text_value = str(value)
    replacements = {
        "φ": "phi",
        "λ": "lambda",
        "²": "2",
        "³": "3",
        "—": "-",
        "–": "-",
        "×": "x",
        "°": " deg",
        "✅": "PASS",
        "⚠️": "WARN",
        "❌": "FAIL",
    }
    for src, dst in replacements.items():
        text_value = text_value.replace(src, dst)
    return text_value.encode("latin-1", "replace").decode("latin-1")


def pdf_add_table(pdf, df: pd.DataFrame, columns: List[str], widths: List[float], max_rows: int = 40) -> None:
    pdf.set_font("Arial", "B", 8)
    for col, width in zip(columns, widths):
        pdf.cell(width, 6, pdf_safe(col), border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 8)
    for _, row in df.head(max_rows).iterrows():
        for col, width in zip(columns, widths):
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt(value, 3 if abs(value) < 10 else 1)
            pdf.cell(width, 6, pdf_safe(value)[:32], border=1)
        pdf.ln()



def make_simple_pdf(lines: List[str]) -> bytes:
    escaped_lines = []
    for line in lines:
        safe = pdf_safe(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        escaped_lines.append(safe[:115])

    stream_parts = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
    first = True
    for line in escaped_lines:
        if not first:
            stream_parts.append("T*")
        stream_parts.append(f"({line}) Tj")
        first = False
    stream_parts.append("ET")
    stream = "\n".join(stream_parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def make_fallback_calculation_pdf(state: DesignState, results: Dict[str, Any]) -> bytes:
    lines = [
        APP_TITLE,
        "Calculation report - preliminary design aid",
        "",
        "Input Summary",
        f"Load case: {state.loadcase.name}",
        f"Pu: {state.loadcase.Pu_kN:.2f} kN",
        f"Mux: {state.loadcase.Mux_kNm:.2f} kN-m",
        f"Muy: {state.loadcase.Muy_kNm:.2f} kN-m",
        f"Vux/Vuy/Tuz: {state.loadcase.Vux_kN:.2f} kN / {state.loadcase.Vuy_kN:.2f} kN / {state.loadcase.Tuz_kNm:.2f} kN-m",
        f"fc': {state.material.fc_MPa:.2f} MPa, fy: {state.material.fy_MPa:.2f} MPa",
        f"Cap: {state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f} mm",
        "",
        "Design Checks",
    ]
    for check in results["checks"]:
        lines.append(f"{check.name}: D={fmt(check.demand, 2)} {check.unit}, C={fmt(check.capacity, 2)}, Ratio={fmt(check.ratio, 3)}, {check.status}")
    lines.extend(["", "Pile Reactions"])
    pile_env_df = results.get("pile_reaction_envelope")
    if isinstance(pile_env_df, pd.DataFrame) and not pile_env_df.empty:
        for _, row in pile_env_df.iterrows():
            lines.append(
                f"{row.get('Pile')}: comp={fmt(safe_float(row.get('Max compression (kN)')), 1)} kN, "
                f"uplift={fmt(safe_float(row.get('Max uplift (kN)')), 1)} kN, "
                f"D/C={fmt(max(safe_float(row.get('Compression ratio')), safe_float(row.get('Tension ratio'))), 3)}"
            )
    else:
        for pile in state.piles:
            lines.append(f"{pile.label}: x={pile.x_mm:.0f} mm, y={pile.y_mm:.0f} mm, R={pile.reaction_kN:.2f} kN")
    lines.extend(
        [
            "",
            "Reinforcement Summary",
            f"X bars: {state.reinforcement.main_bar_x} @ {results['spacing_x']['spacing_use_mm']:.0f} mm, As req {results['flex_x']['As_req_mm2']:.0f} mm2",
            f"Y bars: {state.reinforcement.main_bar_y} @ {results['spacing_y']['spacing_use_mm']:.0f} mm, As req {results['flex_y']['As_req_mm2']:.0f} mm2",
            f"Top X/Y bars: {state.reinforcement.top_bar} @ {results['top_spacing_x']['spacing_use_mm']:.0f} mm, As req X/Y {results['top_flex_x']['As_req_mm2']:.0f}/{results['top_flex_y']['As_req_mm2']:.0f} mm2",
        ]
    )
    return make_simple_pdf(lines)


def make_calculation_pdf(state: DesignState, results: Dict[str, Any]) -> bytes:
    if FPDF is None:
        return make_fallback_calculation_pdf(state, results)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=8)
    pdf.set_margins(10, 8, 10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 6, pdf_safe(APP_TITLE), ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.cell(0, 4, pdf_safe("Calculation summary - verify final design against governing code, geotechnical report, and project specifications."), ln=True)
    pdf.ln(1)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 5, "Input Summary", ln=True)
    pdf_add_table(
        pdf,
        pd.DataFrame(
            [
                {"Item": "Load case", "Value": state.loadcase.name, "Unit": ""},
                {"Item": "Pu", "Value": state.loadcase.Pu_kN, "Unit": "kN"},
                {"Item": "Mux", "Value": state.loadcase.Mux_kNm, "Unit": "kN-m"},
                {"Item": "Muy", "Value": state.loadcase.Muy_kNm, "Unit": "kN-m"},
                {"Item": "Vux", "Value": state.loadcase.Vux_kN, "Unit": "kN"},
                {"Item": "Vuy", "Value": state.loadcase.Vuy_kN, "Unit": "kN"},
                {"Item": "Tuz", "Value": state.loadcase.Tuz_kNm, "Unit": "kN-m"},
                {"Item": "fc'", "Value": state.material.fc_MPa, "Unit": "MPa"},
                {"Item": "fy", "Value": state.material.fy_MPa, "Unit": "MPa"},
                {"Item": "Cap size", "Value": f"{state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f}", "Unit": "mm"},
                {"Item": "Pile count", "Value": state.geometry.n_piles, "Unit": ""},
                {"Item": "Pile diameter", "Value": state.geometry.pile_diameter_mm, "Unit": "mm"},
            ]
        ),
        ["Item", "Value", "Unit"],
        [55, 85, 30],
    )
    pdf.ln(1)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 5, "Design Checks", ln=True)
    checks_df = checks_to_dataframe(results["checks"])
    pdf_add_table(pdf, checks_df, ["Check", "Demand", "Capacity", "Ratio", "Status"], [58, 30, 30, 25, 25])
    pdf.ln(1)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 5, "Pile Reactions - CS Service Envelope", ln=True)
    pile_env_df = results.get("pile_reaction_envelope")
    if isinstance(pile_env_df, pd.DataFrame) and not pile_env_df.empty:
        pile_df = pile_env_df.copy()
        pile_df["Status"] = pile_df.apply(
            lambda row: "FAIL" if max(safe_float(row.get("Compression ratio")), safe_float(row.get("Tension ratio"))) > 1.0 else "PASS",
            axis=1,
        )
        pdf_add_table(pdf, pile_df, ["Pile", "x (mm)", "y (mm)", "Max compression (kN)", "Max uplift (kN)", "Status"], [16, 23, 23, 43, 35, 24])
    else:
        pile_df = pile_group_result_table(state.piles, state.geometry)
        pile_df["Status"] = pile_df.apply(
            lambda row: "FAIL" if max(safe_float(row.get("Compression ratio")), safe_float(row.get("Tension ratio"))) > 1.0 else "PASS",
            axis=1,
        )
        pdf_add_table(pdf, pile_df, ["Pile", "x (mm)", "y (mm)", "R, compression + (kN)", "Status"], [22, 28, 28, 55, 35])
    pdf.ln(1)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 5, "Reinforcement Summary", ln=True)
    reinf_df = pd.DataFrame(
        [
            {"Direction": "Bottom X bars", "Bar": state.reinforcement.main_bar_x, "As req": results["flex_x"]["As_req_mm2"], "Use spacing": results["spacing_x"]["spacing_use_mm"], "As prov": results["spacing_x"]["As_prov_mm2"], "phiMn": results["cap_xbars"]["phiMn_kNm"]},
            {"Direction": "Bottom Y bars", "Bar": state.reinforcement.main_bar_y, "As req": results["flex_y"]["As_req_mm2"], "Use spacing": results["spacing_y"]["spacing_use_mm"], "As prov": results["spacing_y"]["As_prov_mm2"], "phiMn": results["cap_ybars"]["phiMn_kNm"]},
            {"Direction": "Top X bars", "Bar": state.reinforcement.top_bar, "As req": results["top_flex_x"]["As_req_mm2"], "Use spacing": results["top_spacing_x"]["spacing_use_mm"], "As prov": results["top_spacing_x"]["As_prov_mm2"], "phiMn": results["top_cap_xbars"]["phiMn_kNm"]},
            {"Direction": "Top Y bars", "Bar": state.reinforcement.top_bar, "As req": results["top_flex_y"]["As_req_mm2"], "Use spacing": results["top_spacing_y"]["spacing_use_mm"], "As prov": results["top_spacing_y"]["As_prov_mm2"], "phiMn": results["top_cap_ybars"]["phiMn_kNm"]},
        ]
    )
    pdf_add_table(pdf, reinf_df, ["Direction", "Bar", "As req", "Use spacing", "As prov", "phiMn"], [30, 22, 28, 30, 28, 28])
    data = pdf.output(dest="S")
    if isinstance(data, str):
        return data.encode("latin-1")
    return bytes(data)

# =============================================================================
# PYSIDE6-FRIENDLY OVERRIDES AND CONVENIENCE HELPERS
# =============================================================================

def status_badge(status: str) -> str:
    s = str(status).upper()
    if s == "PASS":
        return "PASS"
    if s == "NEAR":
        return "NEAR"
    if s == "FAIL":
        return "FAIL"
    if s == "STM":
        return "STM REVIEW"
    return s


def read_uploaded_table(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = str(getattr(uploaded_file, "name", uploaded_file)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file, header=None)
        return normalize_sap2000_joint_reactions_table(raw)
    if name.endswith(".xlsx"):
        raw = pd.read_excel(uploaded_file, header=None)
        return normalize_sap2000_joint_reactions_table(raw)
    raise ValueError("Unsupported file type. Please use CSV or XLSX.")


def read_sap2000_table_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
    else:
        raise ValueError("Unsupported file type. Please use CSV or XLSX.")
    return normalize_sap2000_joint_reactions_table(raw)


def make_saved_state_xlsx(
    state: DesignState,
    results: Dict[str, Any],
    include_self_weight: bool,
    sap_df: Optional[pd.DataFrame] = None,
    groups_df: Optional[pd.DataFrame] = None,
    batch_design: Optional[Dict[str, Any]] = None,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        key_value_dataframe({"version": 1, "include_self_weight": include_self_weight}).to_excel(writer, sheet_name="settings", index=False)
        key_value_dataframe(asdict(state.material)).to_excel(writer, sheet_name="material", index=False)
        key_value_dataframe(asdict(state.geometry)).to_excel(writer, sheet_name="geometry", index=False)
        key_value_dataframe(asdict(state.reinforcement)).to_excel(writer, sheet_name="reinforcement", index=False)
        key_value_dataframe(asdict(state.loadcase)).to_excel(writer, sheet_name="loadcase", index=False)
        piles_to_dataframe(state.piles).to_excel(writer, sheet_name="piles", index=False)
        checks_to_dataframe(results["checks"]).to_excel(writer, sheet_name="checks", index=False)
        pile_group_result_table(state.piles, state.geometry).to_excel(writer, sheet_name="pile_reactions", index=False)
        pd.DataFrame(
            [
                {"Item": "Cap length X", "Value": state.cap_length_x_mm, "Unit": "mm"},
                {"Item": "Cap width Y", "Value": state.cap_width_y_mm, "Unit": "mm"},
                {"Item": "Self weight", "Value": state.self_weight_kN, "Unit": "kN"},
                {"Item": "Effective depth X", "Value": results["d_x_mm"], "Unit": "mm"},
                {"Item": "Effective depth Y", "Value": results["d_y_mm"], "Unit": "mm"},
                {"Item": "Pu total", "Value": results["Pu_total_kN"], "Unit": "kN"},
                {"Item": "X bars required As", "Value": results["flex_x"]["As_req_mm2"], "Unit": "mm2"},
                {"Item": "Y bars required As", "Value": results["flex_y"]["As_req_mm2"], "Unit": "mm2"},
                {"Item": "Top X bars required As", "Value": results["top_flex_x"]["As_req_mm2"], "Unit": "mm2"},
                {"Item": "Top Y bars required As", "Value": results["top_flex_y"]["As_req_mm2"], "Unit": "mm2"},
            ]
        ).to_excel(writer, sheet_name="summary", index=False)
        if isinstance(sap_df, pd.DataFrame) and not sap_df.empty:
            sap_df.to_excel(writer, sheet_name="sap2000_import", index=False)
        if isinstance(groups_df, pd.DataFrame) and not groups_df.empty:
            groups_df.to_excel(writer, sheet_name="groups", index=False)
            group_rows = []
            if "Group Name" in groups_df.columns and "Joint IDs" in groups_df.columns:
                for _, row in groups_df.iterrows():
                    for joint in parse_joint_list(row.get("Joint IDs", "")):
                        group_rows.append({"Group Name": row.get("Group Name", ""), "Joint": joint})
            pd.DataFrame(group_rows).to_excel(writer, sheet_name="group_joints", index=False)
        if isinstance(batch_design, dict):
            batch_design.get("summary", pd.DataFrame()).to_excel(writer, sheet_name="design_results_summary", index=False)
            batch_design.get("pile_envelopes", pd.DataFrame()).to_excel(writer, sheet_name="pile_reaction_envelopes", index=False)
            batch_design.get("rc_pile_envelopes", pd.DataFrame()).to_excel(writer, sheet_name="rc_pile_reaction_envelopes", index=False)
    return buf.getvalue()


def read_saved_state_xlsx(uploaded_file) -> Optional[Dict[str, Any]]:
    if uploaded_file is None:
        return None
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    payload = {
        "settings": dataframe_to_key_values(sheets.get("settings", pd.DataFrame())),
        "material": dataframe_to_key_values(sheets.get("material", pd.DataFrame())),
        "geometry": dataframe_to_key_values(sheets.get("geometry", pd.DataFrame())),
        "reinforcement": dataframe_to_key_values(sheets.get("reinforcement", pd.DataFrame())),
        "loadcase": dataframe_to_key_values(sheets.get("loadcase", pd.DataFrame())),
        "piles": sheets.get("piles", pd.DataFrame()),
        "sap2000_import": sheets.get("sap2000_import", pd.DataFrame()),
        "groups": sheets.get("groups", pd.DataFrame()),
        "group_joints": sheets.get("group_joints", pd.DataFrame()),
        "design_results_summary": sheets.get("design_results_summary", pd.DataFrame()),
        "pile_reaction_envelopes": sheets.get("pile_reaction_envelopes", pd.DataFrame()),
    }
    if payload["geometry"] or payload["material"] or not payload["piles"].empty:
        return payload
    raise ValueError("Saved state workbook is missing expected sheets.")


def material_from_payload(payload: Dict[str, Any]) -> Material:
    data = payload.get("material", {}) if isinstance(payload, dict) else {}
    return Material(**{field: safe_float(data.get(field, getattr(Material(), field)), getattr(Material(), field)) for field in asdict(Material()).keys()})


def geometry_from_payload(payload: Dict[str, Any]) -> Geometry:
    data = payload.get("geometry", {}) if isinstance(payload, dict) else {}
    geom = Geometry()
    values = {}
    for field, default in asdict(geom).items():
        value = data.get(field, default)
        if isinstance(default, bool):
            values[field] = coerce_bool(value, default)
        elif isinstance(default, int):
            values[field] = int(round(safe_float(value, float(default))))
        elif isinstance(default, float):
            values[field] = safe_float(value, default)
        else:
            values[field] = str(value) if value is not None else default
    return Geometry(**values)


def reinforcement_from_payload(payload: Dict[str, Any]) -> Reinforcement:
    data = payload.get("reinforcement", {}) if isinstance(payload, dict) else {}
    reinf = Reinforcement()
    values = {}
    for field, default in asdict(reinf).items():
        value = data.get(field, default)
        if isinstance(default, float):
            values[field] = safe_float(value, default)
        else:
            values[field] = str(value) if value is not None else default
    return Reinforcement(**values)


def include_self_weight_from_payload(payload: Dict[str, Any], default: bool = True) -> bool:
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    return coerce_bool(settings.get("include_self_weight", default), default)


def design_batch_from_manual_table(
    groups: pd.DataFrame,
    mat: Material,
    base_geom: Geometry,
    reinf: Reinforcement,
    include_self_weight: bool,
) -> Dict[str, Any]:
    groups = normalize_manual_service_load_columns(ensure_group_columns(groups, base_geom))
    batch_items = []
    summary_rows = []
    pile_env_tables = []
    rc_pile_env_tables = []
    pile_combo_tables = []
    rc_pile_combo_tables = []
    for idx, row in groups.iterrows():
        group_name = str(row.get("Group Name", f"Foundation {idx + 1}")).strip() or f"Foundation {idx + 1}"
        geom = geometry_from_group_row(base_geom, row)
        piles = piles_from_group_geometry(geom)
        service, ultimate = manual_service_ultimate_loadcases(row, group_name)
        item = envelope_group_design(
            group_name,
            [service, ultimate],
            mat,
            geom,
            reinf,
            piles,
            include_self_weight,
            service_loadcases=[service],
            ultimate_loadcases=[ultimate],
        )
        batch_items.append(item)
        summary_rows.append(item["summary"])
        pile_env_tables.append(item["pile_envelope"])
        rc_pile_env_tables.append(item["rc_pile_envelope"])
        pile_combo_tables.append(item["results"].get("pile_reactions_by_combo", pd.DataFrame()))
        rc_pile_combo_tables.append(item["results"].get("rc_pile_reactions_by_combo", pd.DataFrame()))
    return {
        "mode": "Manual input",
        "items": batch_items,
        "summary": pd.DataFrame(summary_rows),
        "pile_envelopes": pd.concat(pile_env_tables, ignore_index=True) if pile_env_tables else pd.DataFrame(),
        "rc_pile_envelopes": pd.concat(rc_pile_env_tables, ignore_index=True) if rc_pile_env_tables else pd.DataFrame(),
        "pile_reactions_by_combo": pd.concat(pile_combo_tables, ignore_index=True) if pile_combo_tables else pd.DataFrame(),
        "rc_pile_reactions_by_combo": pd.concat(rc_pile_combo_tables, ignore_index=True) if rc_pile_combo_tables else pd.DataFrame(),
        "groups": groups,
    }


def design_batch_from_sap_table(
    groups: pd.DataFrame,
    sap_df: pd.DataFrame,
    mat: Material,
    base_geom: Geometry,
    reinf: Reinforcement,
    include_self_weight: bool,
    vertical_col: str,
    mx_col: str,
    my_col: str,
    vx_col: Optional[str] = None,
    vy_col: Optional[str] = None,
    torsion_col: Optional[str] = None,
    vertical_sign: float = 1.0,
    moment_sign: float = 1.0,
    selected_case_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if sap_df is None or sap_df.empty:
        raise ValueError("SAP2000 table is empty.")
    cmap = sap2000_column_map(sap_df)
    groups = ensure_group_columns(groups, base_geom)
    case_type_col = cmap.get("case_type")
    joint_col = cmap.get("joint")
    work_df = sap_df.copy()
    if selected_case_types and case_type_col and case_type_col in work_df.columns:
        allowed = {str(v) for v in selected_case_types}
        work_df = work_df[work_df[case_type_col].astype(str).isin(allowed)].copy()
    batch_items = []
    summary_rows = []
    pile_env_tables = []
    rc_pile_env_tables = []
    pile_combo_tables = []
    rc_pile_combo_tables = []
    for idx, row in groups.iterrows():
        group_name = str(row.get("Group Name", f"Foundation {idx + 1}")).strip() or f"Foundation {idx + 1}"
        geom = geometry_from_group_row(base_geom, row)
        piles = piles_from_group_geometry(geom)
        group_df = work_df
        joints = parse_joint_list(row.get("Joint IDs", ""))
        if joints and joint_col and joint_col in group_df.columns:
            joint_set = {str(j).strip() for j in joints}
            group_df = group_df[group_df[joint_col].astype(str).str.strip().isin(joint_set)].copy()
        loadcases = sap_loadcases_from_rows(
            group_df,
            cmap,
            vertical_col,
            mx_col,
            my_col,
            vx_col,
            vy_col,
            torsion_col,
            vertical_sign,
            moment_sign,
            group_name,
        )
        if not loadcases:
            continue
        service_cases = [lc for lc in loadcases if is_service_loadcase(lc)] or loadcases
        ultimate_cases = [lc for lc in loadcases if is_ultimate_loadcase(lc)] or loadcases
        item = envelope_group_design(
            group_name,
            loadcases,
            mat,
            geom,
            reinf,
            piles,
            include_self_weight,
            service_loadcases=service_cases,
            ultimate_loadcases=ultimate_cases,
        )
        batch_items.append(item)
        summary_rows.append(item["summary"])
        pile_env_tables.append(item["pile_envelope"])
        rc_pile_env_tables.append(item["rc_pile_envelope"])
        pile_combo_tables.append(item["results"].get("pile_reactions_by_combo", pd.DataFrame()))
        rc_pile_combo_tables.append(item["results"].get("rc_pile_reactions_by_combo", pd.DataFrame()))
    if not batch_items:
        raise ValueError("No SAP rows matched the foundation groups and filters.")
    return {
        "mode": "SAP2000 joint reaction import",
        "items": batch_items,
        "summary": pd.DataFrame(summary_rows),
        "pile_envelopes": pd.concat(pile_env_tables, ignore_index=True) if pile_env_tables else pd.DataFrame(),
        "rc_pile_envelopes": pd.concat(rc_pile_env_tables, ignore_index=True) if rc_pile_env_tables else pd.DataFrame(),
        "pile_reactions_by_combo": pd.concat(pile_combo_tables, ignore_index=True) if pile_combo_tables else pd.DataFrame(),
        "rc_pile_reactions_by_combo": pd.concat(rc_pile_combo_tables, ignore_index=True) if rc_pile_combo_tables else pd.DataFrame(),
        "groups": groups,
        "sap_df": sap_df,
        "sap_cmap": cmap,
    }

