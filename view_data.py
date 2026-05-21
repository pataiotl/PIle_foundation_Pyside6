from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

import pile_engine as eng


def metric_summary(state: eng.DesignState, results: Dict[str, Any]) -> Dict[str, str]:
    checks = results.get("checks", [])
    max_ratio = max([c.ratio for c in checks if np.isfinite(c.ratio)] + [0.0])
    fails = sum(1 for c in checks if c.status == "FAIL")
    near = sum(1 for c in checks if c.status in {"NEAR", "STM"})
    status = "FAIL" if fails or max_ratio > 1.0 else ("NEAR" if near else "PASS")
    max_pile = max([p.reaction_kN for p in state.piles] + [0.0])
    return {
        "Overall max D/C": f"{max_ratio:.2f}",
        "Status": status,
        "Cap size X x Y": f"{state.cap_length_x_mm:,.0f} x {state.cap_width_y_mm:,.0f} mm",
        "Thickness h": f"{state.geometry.cap_thickness_mm:,.0f} mm",
        "Pu used": f"{results['Pu_total_kN']:,.0f} kN",
        "Max pile R": f"{max_pile:,.0f} kN",
    }


def checks_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    df = eng.checks_to_dataframe(results.get("checks", []))
    if "Ratio" in df.columns:
        df["Ratio"] = pd.to_numeric(df["Ratio"], errors="coerce")
    return df


def flexural_summary_dataframe(state: eng.DesignState, results: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Layer": "Bottom",
            "Direction": "X bars",
            "Bar": state.reinforcement.main_bar_x,
            "Required As (mm2)": results["flex_x"]["As_req_mm2"],
            "Strength As (mm2)": results["flex_x"]["As_strength_mm2"],
            "Minimum As (mm2)": results["flex_x"]["As_min_mm2"],
            "Use spacing (mm)": results["spacing_x"]["spacing_use_mm"],
            "Bars count": results["spacing_x"]["n_bars"],
            "Provided As (mm2)": results["spacing_x"]["As_prov_mm2"],
            "phiMn (kN-m)": results["cap_xbars"]["phiMn_kNm"],
            "D/C": results["moment_demands"]["M_for_X_bars_kNm"] / max(results["cap_xbars"]["phiMn_kNm"], 1e-9),
        },
        {
            "Layer": "Bottom",
            "Direction": "Y bars",
            "Bar": state.reinforcement.main_bar_y,
            "Required As (mm2)": results["flex_y"]["As_req_mm2"],
            "Strength As (mm2)": results["flex_y"]["As_strength_mm2"],
            "Minimum As (mm2)": results["flex_y"]["As_min_mm2"],
            "Use spacing (mm)": results["spacing_y"]["spacing_use_mm"],
            "Bars count": results["spacing_y"]["n_bars"],
            "Provided As (mm2)": results["spacing_y"]["As_prov_mm2"],
            "phiMn (kN-m)": results["cap_ybars"]["phiMn_kNm"],
            "D/C": results["moment_demands"]["M_for_Y_bars_kNm"] / max(results["cap_ybars"]["phiMn_kNm"], 1e-9),
        },
        {
            "Layer": "Top",
            "Direction": "X bars",
            "Bar": state.reinforcement.top_bar,
            "Required As (mm2)": results["top_flex_x"]["As_req_mm2"],
            "Strength As (mm2)": results["top_flex_x"]["As_strength_mm2"],
            "Minimum As (mm2)": results["top_flex_x"]["As_min_mm2"],
            "Use spacing (mm)": results["top_spacing_x"]["spacing_use_mm"],
            "Bars count": results["top_spacing_x"]["n_bars"],
            "Provided As (mm2)": results["top_spacing_x"]["As_prov_mm2"],
            "phiMn (kN-m)": results["top_cap_xbars"]["phiMn_kNm"],
            "D/C": results["top_demand_x"]["demand_kNm"] / max(results["top_cap_xbars"]["phiMn_kNm"], 1e-9)
            if results["top_demand_x"]["demand_kNm"] > 1e-9 else np.nan,
        },
        {
            "Layer": "Top",
            "Direction": "Y bars",
            "Bar": state.reinforcement.top_bar,
            "Required As (mm2)": results["top_flex_y"]["As_req_mm2"],
            "Strength As (mm2)": results["top_flex_y"]["As_strength_mm2"],
            "Minimum As (mm2)": results["top_flex_y"]["As_min_mm2"],
            "Use spacing (mm)": results["top_spacing_y"]["spacing_use_mm"],
            "Bars count": results["top_spacing_y"]["n_bars"],
            "Provided As (mm2)": results["top_spacing_y"]["As_prov_mm2"],
            "phiMn (kN-m)": results["top_cap_ybars"]["phiMn_kNm"],
            "D/C": results["top_demand_y"]["demand_kNm"] / max(results["top_cap_ybars"]["phiMn_kNm"], 1e-9)
            if results["top_demand_y"]["demand_kNm"] > 1e-9 else np.nan,
        },
    ]
    return pd.DataFrame(rows)


def detail_summary_dataframe(state: eng.DesignState, results: Dict[str, Any]) -> pd.DataFrame:
    mto = eng.calculate_material_takeoff(state, results)
    rows = [
        ("Concrete Volume", mto["concrete_vol_m3"], "m3"),
        ("Main Rebar Weight", mto["main_rebar_kg"], "kg"),
        ("Rebar Ratio", mto["rebar_ratio_kg_m3"], "kg/m3"),
        ("Effective depth X bars", results["d_x_mm"], "mm"),
        ("Effective depth Y bars", results["d_y_mm"], "mm"),
        ("Effective depth top X bars", results["d_top_x_mm"], "mm"),
        ("Effective depth top Y bars", results["d_top_y_mm"], "mm"),
        ("Top governing moment, X bars", results["top_demand_x"]["demand_kNm"], "kN-m"),
        ("Top governing moment, Y bars", results["top_demand_y"]["demand_kNm"], "kN-m"),
        ("Top net uplift moment, X bars", results["top_demand_x"]["uplift_tension_kNm"], "kN-m"),
        ("Top net uplift moment, Y bars", results["top_demand_y"]["uplift_tension_kNm"], "kN-m"),
        ("Top continuous M-, X strip", results["top_demand_x"]["continuous_negative_kNm"], "kN-m"),
        ("Top continuous M-, Y strip", results["top_demand_y"]["continuous_negative_kNm"], "kN-m"),
        ("Equivalent w, X strip", results["continuous_x"]["equivalent_w_kN_per_m"], "kN/m"),
        ("Equivalent w, Y strip", results["continuous_y"]["equivalent_w_kN_per_m"], "kN/m"),
        ("ACI 318-25 close-pile provision", "YES" if results["close_pile_spacing"]["applies"] else "NO", ""),
        ("Max nearest pile spacing", results["close_pile_spacing"]["max_nearest_spacing_mm"], "mm"),
        ("4D close-pile limit", results["close_pile_spacing"]["limit_mm"], "mm"),
        ("One-way phiVc, section normal to Y", results["one_way_x"]["phiVc_kN"], "kN"),
        ("One-way phiVc, section normal to X", results["one_way_y"]["phiVc_kN"], "kN"),
        ("Punching bo", results["punch_demand"]["bo_mm"], "mm"),
        ("Punching phiVc", results["punch_cap"]["phiVc_kN"], "kN"),
        ("Punching beta", results["punch_cap"]["beta"], ""),
        ("Punching alpha_s", results["punch_cap"]["alpha_s"], ""),
        ("Development estimate X bars", results["ld_x_mm"], "mm"),
        ("Development estimate Y bars", results["ld_y_mm"], "mm"),
        ("Development estimate top bars", results["ld_top_mm"], "mm"),
        ("STM tie As advisory X bars", results["As_stm_x_mm2"], "mm2"),
        ("STM tie As advisory Y bars", results["As_stm_y_mm2"], "mm2"),
    ]
    return pd.DataFrame([{"Item": item, "Value": value, "Unit": unit} for item, value, unit in rows])


def governing_combos_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    combos = results.get("governing_combos", {})
    return pd.DataFrame([{"Item": key, "Combination": value} for key, value in combos.items()])


def engineering_notes(results: Dict[str, Any]) -> List[str]:
    notes = []
    stm = results.get("stm", pd.DataFrame())
    if any(c.status == "STM" for c in results.get("checks", [])):
        notes.append(
            "One or more sectional shear checks are not meaningful because the critical section encloses the pile reactions. Verify STM struts, ties, nodal zones, and anchorage."
        )
    if isinstance(stm, pd.DataFrame) and not stm.empty and "theta (deg)" in stm.columns:
        if (stm["theta (deg)"] < 25.0).any():
            notes.append("Some STM strut angles are below about 25 deg. A deeper cap or revised pile spacing may be required.")
        if (stm["theta (deg)"] > 65.0).any():
            notes.append("Some STM strut angles are steep. Verify nodal zone geometry, anchorage, and local bearing.")
    if results["punch_demand"]["R_inside_kN"] <= 0:
        notes.append("No pile reaction is inside the punching perimeter. Punching demand may be severe; verify critical section logic.")
    if results["flex_x"]["rho"] > 0.02 or results["flex_y"]["rho"] > 0.02:
        notes.append("Flexural reinforcement ratio is high. Increase cap thickness or use a full STM.")
    if results["As_stm_x_mm2"] > results["spacing_x"]["As_prov_mm2"]:
        notes.append("STM advisory tie area for X bars exceeds provided X reinforcement. Add ties or revise the cap layout.")
    if results["As_stm_y_mm2"] > results["spacing_y"]["As_prov_mm2"]:
        notes.append("STM advisory tie area for Y bars exceeds provided Y reinforcement. Add ties or revise the cap layout.")
    return notes or ["No major engineering advisory flags were generated for the selected result."]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
