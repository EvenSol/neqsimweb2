"""
Lab / LIMS composition import — parse lab sample data and update feed streams.

Accepts CSV or JSON composition data (from LIMS export, lab analysis, or
manual entry) and:
  1. Maps component names to NeqSim naming convention
  2. Normalises compositions to mole fraction
  3. Compares old vs new composition (delta table)
  4. Optionally applies the new composition to a feed stream in the model
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Component name mapping — common lab / LIMS aliases → NeqSim names
# ---------------------------------------------------------------------------

_ALIASES: Dict[str, str] = {
    # Hydrocarbons
    "c1": "methane", "ch4": "methane", "methane": "methane",
    "c2": "ethane", "c2h6": "ethane", "ethane": "ethane",
    "c3": "propane", "c3h8": "propane", "propane": "propane",
    "ic4": "i-butane", "i-c4": "i-butane", "isobutane": "i-butane",
    "i-butane": "i-butane",
    "nc4": "n-butane", "n-c4": "n-butane", "n-butane": "n-butane",
    "butane": "n-butane",
    "ic5": "i-pentane", "i-c5": "i-pentane", "isopentane": "i-pentane",
    "i-pentane": "i-pentane",
    "nc5": "n-pentane", "n-c5": "n-pentane", "n-pentane": "n-pentane",
    "pentane": "n-pentane",
    "nc6": "n-hexane", "n-c6": "n-hexane", "hexane": "n-hexane",
    "n-hexane": "n-hexane",
    "nc7": "n-heptane", "n-c7": "n-heptane", "heptane": "n-heptane",
    "n-heptane": "n-heptane",
    "nc8": "n-octane", "n-c8": "n-octane", "octane": "n-octane",
    "n-octane": "n-octane",
    # Plus fractions
    "c7": "C7", "c7+": "C7", "c8": "C8", "c9": "C9", "c10": "C10",
    "c11": "C11", "c12": "C12", "c13": "C13", "c14": "C14",
    "c15": "C15", "c16": "C16", "c17": "C17", "c18": "C18",
    "c19": "C19", "c20": "C20",
    # Acid gases
    "co2": "CO2", "carbon dioxide": "CO2",
    "h2s": "H2S", "hydrogen sulfide": "H2S", "hydrogen sulphide": "H2S",
    # Inerts
    "n2": "nitrogen", "nitrogen": "nitrogen",
    "o2": "oxygen", "oxygen": "oxygen",
    "h2": "hydrogen", "hydrogen": "hydrogen",
    "he": "helium", "helium": "helium",
    "ar": "argon", "argon": "argon",
    # Water & inhibitors
    "h2o": "water", "water": "water",
    "meoh": "methanol", "methanol": "methanol",
    "meg": "MEG", "eg": "MEG", "ethylene glycol": "MEG",
    "teg": "TEG", "triethylene glycol": "TEG",
    "deg": "DEG", "diethylene glycol": "DEG",
}


def _normalise_name(raw: str) -> str:
    """Map a lab component name to NeqSim convention."""
    key = raw.strip().lower().replace("_", "-")
    return _ALIASES.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComponentDelta:
    """Change in one component's composition."""
    component: str
    old_mole_frac: float
    new_mole_frac: float
    delta_mole_frac: float
    delta_pct: float  # relative % change


@dataclass
class LabSample:
    """Parsed lab sample."""
    sample_id: str
    source: str  # "csv", "json", or "inline"
    timestamp: Optional[str]
    components: Dict[str, float]  # NeqSim-name → mole fraction (normalised)
    unmapped: List[str]  # component names that couldn't be mapped
    raw_total: float  # sum before normalisation


@dataclass
class LabImportResult:
    """Top-level result returned to the chat layer."""
    sample: LabSample
    stream_name: str  # which stream was (or would be) updated
    applied: bool  # True if actually applied to the model
    deltas: List[ComponentDelta]
    warnings: List[str]
    composition_df: Optional[Any] = None  # pd.DataFrame for display


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_csv_composition(csv_text: str, sample_id: str = "LAB-001") -> LabSample:
    """Parse CSV text with columns like ComponentName, MoleFraction.

    Accepts various header names; first column is always component name,
    second column is the numeric composition value.
    """
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV data.")

    # Detect header
    header = [h.strip().lower() for h in rows[0]]
    has_header = any(kw in header[0] for kw in ("component", "name", "species"))
    data_rows = rows[1:] if has_header else rows

    components: Dict[str, float] = {}
    unmapped: List[str] = []
    for row in data_rows:
        if len(row) < 2:
            continue
        raw_name = row[0].strip()
        if not raw_name:
            continue
        try:
            value = float(row[1].strip().replace(",", "."))
        except ValueError:
            continue
        neqsim_name = _normalise_name(raw_name)
        if neqsim_name == raw_name and raw_name.lower() not in _ALIASES:
            unmapped.append(raw_name)
        components[neqsim_name] = components.get(neqsim_name, 0.0) + value

    return _finalise_sample(components, unmapped, sample_id, "csv")


def parse_json_composition(json_text: str, sample_id: str = "LAB-001") -> LabSample:
    """Parse JSON composition data.

    Accepted formats:
      {"components": {"CH4": 0.85, "C2H6": 0.07, ...}}
      [{"name": "CH4", "mole_fraction": 0.85}, ...]
      {"CH4": 0.85, "C2H6": 0.07, ...}  (flat dict)
    """
    data = json.loads(json_text.strip())
    raw_dict: Dict[str, float] = {}

    if isinstance(data, dict):
        if "components" in data:
            inner = data["components"]
            sample_id = data.get("sample_id", sample_id)
        else:
            inner = data
        if isinstance(inner, dict):
            raw_dict = {k: float(v) for k, v in inner.items()}
        elif isinstance(inner, list):
            for item in inner:
                name_key = next((k for k in item if k.lower() in ("name", "component")), None)
                val_key = next((k for k in item if k.lower() in ("mole_fraction", "molefraction", "value", "fraction")), None)
                if name_key and val_key:
                    raw_dict[item[name_key]] = float(item[val_key])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name_key = next((k for k in item if k.lower() in ("name", "component")), None)
                val_key = next((k for k in item if k.lower() in ("mole_fraction", "molefraction", "value", "fraction")), None)
                if name_key and val_key:
                    raw_dict[item[name_key]] = float(item[val_key])

    if not raw_dict:
        raise ValueError("Could not parse composition from JSON data.")

    components: Dict[str, float] = {}
    unmapped: List[str] = []
    for raw_name, value in raw_dict.items():
        neqsim_name = _normalise_name(raw_name)
        if neqsim_name == raw_name and raw_name.lower() not in _ALIASES:
            unmapped.append(raw_name)
        components[neqsim_name] = components.get(neqsim_name, 0.0) + value

    return _finalise_sample(components, unmapped, sample_id, "json")


def parse_inline_composition(comp_dict: Dict[str, float], sample_id: str = "LAB-001") -> LabSample:
    """Parse an inline composition dict (from LLM structured output)."""
    components: Dict[str, float] = {}
    unmapped: List[str] = []
    for raw_name, value in comp_dict.items():
        neqsim_name = _normalise_name(raw_name)
        if neqsim_name == raw_name and raw_name.lower() not in _ALIASES:
            unmapped.append(raw_name)
        components[neqsim_name] = components.get(neqsim_name, 0.0) + float(value)

    return _finalise_sample(components, unmapped, sample_id, "inline")


def _finalise_sample(
    components: Dict[str, float],
    unmapped: List[str],
    sample_id: str,
    source: str,
) -> LabSample:
    """Normalise to mole fractions summing to 1.0."""
    raw_total = sum(components.values())

    # Detect if values are percentages (sum > 2 implies %)
    if raw_total > 2.0:
        components = {k: v / 100.0 for k, v in components.items()}
        raw_total = sum(components.values())

    # Normalise
    if raw_total > 0:
        components = {k: v / raw_total for k, v in components.items()}

    return LabSample(
        sample_id=sample_id,
        source=source,
        timestamp=None,
        components=components,
        unmapped=unmapped,
        raw_total=raw_total,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lab_import(
    model: Any = None,
    csv_text: Optional[str] = None,
    json_text: Optional[str] = None,
    inline_comp: Optional[Dict[str, float]] = None,
    stream_name: Optional[str] = None,
    sample_id: str = "LAB-001",
    apply_to_model: bool = False,
) -> LabImportResult:
    """Parse lab composition and optionally apply to a feed stream.

    Parameters
    ----------
    model : NeqSimProcessModel, optional
        Process model to update.
    csv_text : str, optional
        Raw CSV text with component-name, mole-fraction columns.
    json_text : str, optional
        Raw JSON text with composition data.
    inline_comp : dict, optional
        Inline component→value dict (from LLM structured output).
    stream_name : str, optional
        Name of the stream to update.  Auto-detects feed stream if omitted.
    sample_id : str
        Sample identifier for traceability.
    apply_to_model : bool
        If True, actually update the model feed stream composition.
    """
    # Parse
    if csv_text:
        sample = parse_csv_composition(csv_text, sample_id)
    elif json_text:
        sample = parse_json_composition(json_text, sample_id)
    elif inline_comp:
        sample = parse_inline_composition(inline_comp, sample_id)
    else:
        raise ValueError("Provide csv_text, json_text, or inline_comp.")

    # Determine target stream
    if not stream_name and model is not None:
        # Auto-detect first feed stream
        streams = model.get_streams_info()
        for s in streams:
            if "feed" in s.name.lower() or "inlet" in s.name.lower():
                stream_name = s.name
                break
        if not stream_name and streams:
            stream_name = streams[0].name

    stream_name = stream_name or "feed"

    # Get old composition for comparison
    old_comp: Dict[str, float] = {}
    if model is not None:
        try:
            summary = model.get_model_summary()
            # Try to find stream in summary
            for line in summary.split("\n"):
                # Heuristic: look through model to get existing composition
                pass
        except Exception:
            pass

        # Try to read composition from the actual Java stream
        try:
            proc_sys = model.get_process_system()
            if proc_sys:
                stream_obj = proc_sys.getMeasurementDevice(stream_name)
                if stream_obj is None:
                    # Try as unit operation
                    for i in range(proc_sys.size()):
                        unit = proc_sys.get(i)
                        if hasattr(unit, 'getName') and unit.getName() == stream_name:
                            if hasattr(unit, 'getFluid'):
                                fluid = unit.getFluid()
                                if fluid:
                                    for j in range(fluid.getNumberOfComponents()):
                                        comp = fluid.getComponent(j)
                                        old_comp[comp.getComponentName()] = comp.getz()
                            break
        except Exception:
            pass

    # Compute deltas
    all_components = set(list(sample.components.keys()) + list(old_comp.keys()))
    deltas: List[ComponentDelta] = []
    for comp in sorted(all_components):
        old_val = old_comp.get(comp, 0.0)
        new_val = sample.components.get(comp, 0.0)
        delta = new_val - old_val
        rel_pct = (delta / old_val * 100) if old_val > 1e-10 else (100.0 if new_val > 0 else 0.0)
        deltas.append(ComponentDelta(
            component=comp,
            old_mole_frac=old_val,
            new_mole_frac=new_val,
            delta_mole_frac=delta,
            delta_pct=rel_pct,
        ))

    # Build display DataFrame
    comp_df = pd.DataFrame([{
        "Component": d.component,
        "Old Mole Frac": round(d.old_mole_frac, 6),
        "New Mole Frac": round(d.new_mole_frac, 6),
        "Delta": round(d.delta_mole_frac, 6),
        "Change %": round(d.delta_pct, 1),
    } for d in deltas if d.new_mole_frac > 0 or d.old_mole_frac > 0])

    # Warnings
    warnings: List[str] = []
    if sample.unmapped:
        warnings.append(f"Unmapped components (used as-is): {', '.join(sample.unmapped)}")
    if abs(sample.raw_total - 1.0) > 0.01:
        warnings.append(f"Raw composition summed to {sample.raw_total:.4f} — normalised to 1.0.")

    # Heavy components check
    heavies = sum(v for k, v in sample.components.items()
                  if k.startswith("C") and k[1:].isdigit() and int(k[1:]) >= 7)
    if heavies > 0.2:
        warnings.append(f"Heavy fraction (C7+) is {heavies:.1%} — verify plus-fraction properties.")

    # Apply to model if requested
    applied = False
    if apply_to_model and model is not None:
        try:
            proc_sys = model.get_process_system()
            if proc_sys:
                for i in range(proc_sys.size()):
                    unit = proc_sys.get(i)
                    if hasattr(unit, 'getName') and unit.getName() == stream_name:
                        if hasattr(unit, 'getFluid'):
                            fluid = unit.getFluid()
                            if fluid:
                                for comp_name, mole_frac in sample.components.items():
                                    try:
                                        idx = fluid.getComponentIndex(comp_name)
                                        if idx >= 0:
                                            fluid.addComponent(comp_name, mole_frac)
                                    except Exception:
                                        pass
                                applied = True
                        break
        except Exception as exc:
            warnings.append(f"Could not apply to model: {exc}")

    return LabImportResult(
        sample=sample,
        stream_name=stream_name,
        applied=applied,
        deltas=deltas,
        warnings=warnings,
        composition_df=comp_df,
    )


# ---------------------------------------------------------------------------
# Formatter for LLM context
# ---------------------------------------------------------------------------

def format_lab_import_result(result: LabImportResult) -> str:
    """Format lab import result for LLM follow-up."""
    lines = [
        f"=== Lab Composition Import ===",
        f"Sample ID:    {result.sample.sample_id}",
        f"Source:        {result.sample.source}",
        f"Stream:       {result.stream_name}",
        f"Applied:      {'Yes' if result.applied else 'No (preview only)'}",
        f"Components:   {len(result.sample.components)}",
    ]

    if result.sample.unmapped:
        lines.append(f"Unmapped:     {', '.join(result.sample.unmapped)}")

    lines.append("")
    lines.append("--- Composition Comparison ---")
    lines.append(f"{'Component':<20} {'Old':>10} {'New':>10} {'Delta':>10} {'Change%':>8}")
    lines.append("-" * 60)
    for d in result.deltas:
        if d.new_mole_frac > 0 or d.old_mole_frac > 0:
            lines.append(
                f"{d.component:<20} {d.old_mole_frac:>10.6f} {d.new_mole_frac:>10.6f} "
                f"{d.delta_mole_frac:>+10.6f} {d.delta_pct:>+7.1f}%"
            )

    if result.warnings:
        lines.append("")
        lines.append("--- Warnings ---")
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)
