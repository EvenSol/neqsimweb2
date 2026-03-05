"""
Signal Tracker — follow stream properties and equipment KPIs across chat runs.

Allows the user to "watch" specific signals (e.g. compressor power, separator
temperature, export gas flow) and accumulate a history of values every time
the model is run.  The history is displayed as a trend table showing how
values change across successive calculations.

Usage from chat:
  - "track compressor power"   → starts watching that signal
  - "track feed gas temperature, export gas flow"
  - "show tracked signals"     → displays current trend table
  - "stop tracking compressor power"
  - "clear tracking"           → resets all history

The tracker is integrated into the chat session — every scenario run,
sensitivity point, or what-if automatically snapshots all watched signals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    """One recorded value of a signal at a point in time."""
    value: Optional[float] = None
    timestamp: float = 0.0          # time.time()
    label: str = ""                 # e.g. "Base", "Scenario 1", "After pressure change"
    run_index: int = 0              # sequential run counter


@dataclass
class TrackedSignal:
    """Definition + history of one tracked signal."""
    signal_id: str                  # e.g. "compressor 1.power_kW"
    display_name: str = ""          # human-friendly name
    unit: str = ""
    history: List[SignalSnapshot] = field(default_factory=list)


@dataclass
class SignalTrackerState:
    """Serialisable state of the tracker."""
    signals: Dict[str, TrackedSignal] = field(default_factory=dict)
    run_counter: int = 0
    labels: List[str] = field(default_factory=list)   # ordered list of run labels


# ---------------------------------------------------------------------------
# Signal resolution — map user language to model KPIs / stream properties
# ---------------------------------------------------------------------------

# Known property aliases — map user words to KPI suffixes.
# These are convenience aliases; any KPI key from the model works as fallback.
_STREAM_PROPS = {
    # conditions
    "temperature_c": ("temperature_C", "°C"),
    "temperature": ("temperature_C", "°C"),
    "temp": ("temperature_C", "°C"),
    "pressure_bara": ("pressure_bara", "bara"),
    "pressure": ("pressure_bara", "bara"),
    "flow_kg_hr": ("flow_kg_hr", "kg/hr"),
    "flow": ("flow_kg_hr", "kg/hr"),
    "flow_rate": ("flow_kg_hr", "kg/hr"),
    "flow rate": ("flow_kg_hr", "kg/hr"),
    # vapor pressures
    "tvp": ("TVP_bara", "bara"),
    "true vapor pressure": ("TVP_bara", "bara"),
    "true vapour pressure": ("TVP_bara", "bara"),
    "rvp": ("RVP_bara", "bara"),
    "reid vapor pressure": ("RVP_bara", "bara"),
    "reid vapour pressure": ("RVP_bara", "bara"),
    # transport / thermo
    "viscosity": ("viscosity_Pa_s", "Pa·s"),
    "kinematic viscosity": ("kinematic_viscosity_m2_s", "m2/s"),
    "thermal conductivity": ("thermal_conductivity_W_mK", "W/(m·K)"),
    "conductivity": ("thermal_conductivity_W_mK", "W/(m·K)"),
    "density": ("density_kg_m3", "kg/m3"),
    "molar mass": ("molar_mass_kg_mol", "kg/mol"),
    "molecular weight": ("molar_mass_kg_mol", "kg/mol"),
    "mw": ("molar_mass_kg_mol", "kg/mol"),
    # thermo properties
    "z factor": ("Z_factor", "[-]"),
    "z-factor": ("Z_factor", "[-]"),
    "compressibility": ("Z_factor", "[-]"),
    "enthalpy": ("enthalpy_J_kg", "J/kg"),
    "entropy": ("entropy_J_kgK", "J/(kg·K)"),
    "cp": ("Cp_kJ_kgK", "kJ/(kg·K)"),
    "heat capacity": ("Cp_kJ_kgK", "kJ/(kg·K)"),
    "cv": ("Cv_kJ_kgK", "kJ/(kg·K)"),
    "jt coefficient": ("JT_coefficient_K_bar", "K/bar"),
    "joule thomson": ("JT_coefficient_K_bar", "K/bar"),
    "joule-thomson": ("JT_coefficient_K_bar", "K/bar"),
    "sound speed": ("sound_speed_m_s", "m/s"),
    "speed of sound": ("sound_speed_m_s", "m/s"),
    # phase
    "gas fraction": ("gas_phase_fraction", "[-]"),
    "oil fraction": ("oil_phase_fraction", "[-]"),
    "water fraction": ("aqueous_phase_fraction", "[-]"),
    "number of phases": ("number_of_phases", "[-]"),
}

_UNIT_PROPS = {
    # generic
    "power_kw": ("power_kW", "kW"),
    "power": ("power_kW", "kW"),
    "duty_kw": ("duty_kW", "kW"),
    "duty": ("duty_kW", "kW"),
    # compressor
    "polytropic head": ("polytropicHead_kJkg", "kJ/kg"),
    "head": ("polytropicHead_kJkg", "kJ/kg"),
    "compression ratio": ("compressionRatio", "[-]"),
    "speed": ("speed_rpm", "rpm"),
    "rpm": ("speed_rpm", "rpm"),
    "distance to surge": ("distanceToSurge", "[-]"),
    "surge margin": ("distanceToSurge", "[-]"),
    "polytropic exponent": ("polytropicExponent", "[-]"),
    "utilization": ("maxUtilizationPercent", "%"),
    # separator
    "gas load factor": ("gasLoadFactor", "m/s"),
    "gas velocity": ("gasSuperficialVelocity", "m/s"),
    "liquid level": ("liquidLevel", "m"),
    "carryover": ("liquidCarryoverFraction", "[-]"),
    "carryunder": ("gasCarryunderFraction", "[-]"),
    # heat exchange
    "pressure drop": ("pressureDrop_bar", "bar"),
    "ua": ("UAvalue", "W/K"),
    "ua value": ("UAvalue", "W/K"),
}


def _alias_matches(alias: str, text: str) -> bool:
    """Check if alias appears in text as a whole word (not as a substring of another word)."""
    import re
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text))


def _resolve_signal(
    model: NeqSimProcessModel,
    user_signal: str,
) -> Tuple[str, str, str]:
    """Resolve a user's signal description to (signal_id, display_name, unit).

    Resolution order:
      1. "total" aggregate KPIs  (total power, total duty)
      2. Stream name + property alias  ("feed gas TVP")
      3. Unit name + property alias  ("compressor 1 surge margin")
      4. Direct KPI-dictionary fuzzy match  (catches anything the model reports)
      5. Raw text fallback
    """
    ul = user_signal.strip().lower()

    # 1. "total" KPIs
    if "total" in ul and "power" in ul:
        return "total_power_kW", "Total Power", "kW"
    if "total" in ul and "duty" in ul:
        return "total_duty_kW", "Total Duty", "kW"

    # 2. Stream properties
    streams = model.list_streams()
    for s in streams:
        sname = s.name.lower()
        if sname in ul:
            remainder = ul.replace(sname, "").strip()
            # Check alias table (longest match first to prefer specific aliases)
            best_match = None
            best_len = 0
            for alias, (prop_id, unit) in _STREAM_PROPS.items():
                if _alias_matches(alias, remainder) or _alias_matches(alias, ul):
                    if len(alias) > best_len:
                        best_match = (prop_id, unit)
                        best_len = len(alias)
            if best_match:
                prop_id, unit = best_match
                kpi_key = f"{s.name}.{prop_id}"
                display = f"{s.name} {prop_id}"
                return kpi_key, display, unit
            # Default to temperature if no property specified
            return f"{s.name}.temperature_C", f"{s.name} temperature", "°C"

    # 3. Unit properties
    units = model.list_units()
    for u in units:
        uname = u.name.lower()
        if uname in ul or any(w in uname for w in ul.split() if len(w) > 2):
            remainder = ul.replace(uname, "").strip()
            best_match = None
            best_len = 0
            for alias, (prop_id, unit) in _UNIT_PROPS.items():
                if _alias_matches(alias, remainder) or _alias_matches(alias, ul):
                    if len(alias) > best_len:
                        best_match = (prop_id, unit)
                        best_len = len(alias)
            if best_match:
                prop_id, unit = best_match
                kpi_key = f"{u.name}.{prop_id}"
                display = f"{u.name} {prop_id}"
                return kpi_key, display, unit
            # Default based on equipment type
            utype = u.unit_type.lower()
            if "compressor" in utype:
                return f"{u.name}.power_kW", f"{u.name} power", "kW"
            elif "cooler" in utype or "heater" in utype:
                return f"{u.name}.duty_kW", f"{u.name} duty", "kW"
            else:
                return f"{u.name}.power_kW", f"{u.name} power", "kW"

    # 4. Fuzzy match against full KPI dictionary
    try:
        rr = model.run()
        kpis = rr.kpis
        best_key = None
        best_score = 0
        for k in kpis:
            words_matched = sum(1 for w in ul.split() if w in k.lower())
            if words_matched > best_score:
                best_score = words_matched
                best_key = k
        if best_key and best_score > 0:
            kpi = kpis[best_key]
            return best_key, best_key, kpi.unit
    except Exception:
        pass

    # 5. Fallback: use the raw text as-is (will fuzzy match at read time)
    return user_signal.strip(), user_signal.strip(), ""


def _read_signal_value(model: NeqSimProcessModel, signal_id: str) -> Optional[float]:
    """Read the current value of a signal from the model.

    Looks up any KPI that the process model reports — stream properties
    (T, P, flow, TVP, RVP, viscosity, density, …), unit properties
    (power, duty, compression ratio, gas load factor, …), totals, and
    anything in the JSON report.
    """
    try:
        rr = model.run()
        kpis = rr.kpis

        # Exact match
        if signal_id in kpis:
            return kpis[signal_id].value

        # Fuzzy / substring match
        sid_lower = signal_id.lower()
        for k, v in kpis.items():
            if sid_lower in k.lower() and v.value is not None:
                return v.value
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Signal Tracker class
# ---------------------------------------------------------------------------

class SignalTracker:
    """Tracks selected signals across multiple model runs."""

    def __init__(self):
        self._state = SignalTrackerState()

    @property
    def signals(self) -> Dict[str, TrackedSignal]:
        return self._state.signals

    @property
    def run_counter(self) -> int:
        return self._state.run_counter

    @property
    def labels(self) -> List[str]:
        return self._state.labels

    def has_signals(self) -> bool:
        return len(self._state.signals) > 0

    def add_signal(
        self,
        model: NeqSimProcessModel,
        user_description: str,
    ) -> Tuple[str, str]:
        """Add a signal to track. Returns (signal_id, display_name)."""
        signal_id, display, unit = _resolve_signal(model, user_description)

        if signal_id not in self._state.signals:
            self._state.signals[signal_id] = TrackedSignal(
                signal_id=signal_id,
                display_name=display,
                unit=unit,
            )

        return signal_id, display

    def remove_signal(self, signal_ref: str) -> bool:
        """Remove a signal by ID or display name (fuzzy)."""
        ref_lower = signal_ref.lower()

        # Exact match
        if signal_ref in self._state.signals:
            del self._state.signals[signal_ref]
            return True

        # Fuzzy match
        to_remove = None
        for sid, sig in self._state.signals.items():
            if ref_lower in sid.lower() or ref_lower in sig.display_name.lower():
                to_remove = sid
                break
        if to_remove:
            del self._state.signals[to_remove]
            return True
        return False

    def clear(self):
        """Remove all tracked signals and history."""
        self._state = SignalTrackerState()

    def snapshot(
        self,
        model: NeqSimProcessModel,
        label: str = "",
    ) -> Dict[str, Optional[float]]:
        """Record the current values of all tracked signals.
        
        Called automatically after each model run. Returns a dict of
        signal_id → current_value.
        """
        if not self._state.signals:
            return {}

        self._state.run_counter += 1
        idx = self._state.run_counter

        if not label:
            label = f"Run {idx}"
        self._state.labels.append(label)

        now = time.time()
        values: Dict[str, Optional[float]] = {}

        for sid, sig in self._state.signals.items():
            val = _read_signal_value(model, sid)
            sig.history.append(SignalSnapshot(
                value=val,
                timestamp=now,
                label=label,
                run_index=idx,
            ))
            values[sid] = val

        return values

    def get_trend_table(self) -> str:
        """Format the full trend history as a text table."""
        if not self._state.signals:
            return "No signals are being tracked."

        signals = list(self._state.signals.values())

        # Find the maximum history length
        max_hist = max(len(s.history) for s in signals) if signals else 0
        if max_hist == 0:
            header_parts = ["Signal"]
            for s in signals:
                unit_str = f" ({s.unit})" if s.unit else ""
                header_parts.append(f"{s.display_name}{unit_str}")
            return "Tracked signals (no snapshots yet):\n" + "\n".join(
                f"  - {s.display_name} ({s.unit})" for s in signals
            )

        # Build table
        lines = ["=== SIGNAL TRACKER ===", ""]

        # Header
        header = f"{'Run':<20}"
        for s in signals:
            unit_str = f"({s.unit})" if s.unit else ""
            col_name = f"{s.display_name} {unit_str}"
            header += f"  {col_name:>20}"
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for i in range(max_hist):
            label = ""
            row_values = []
            for s in signals:
                if i < len(s.history):
                    snap = s.history[i]
                    if not label:
                        label = snap.label
                    row_values.append(snap.value)
                else:
                    row_values.append(None)

            row = f"{label:<20}"
            for val in row_values:
                if val is not None:
                    row += f"  {val:>20.4f}"
                else:
                    row += f"  {'N/A':>20}"
            lines.append(row)

        # Delta summary (first → last)
        if max_hist >= 2:
            lines.append("")
            lines.append("CHANGE (first → last):")
            for s in signals:
                first_val = s.history[0].value
                last_val = s.history[-1].value
                if first_val is not None and last_val is not None:
                    delta = last_val - first_val
                    if first_val != 0:
                        pct = delta / abs(first_val) * 100
                        lines.append(f"  {s.display_name}: {first_val:.4f} → {last_val:.4f} ({delta:+.4f}, {pct:+.1f}%)")
                    else:
                        lines.append(f"  {s.display_name}: {first_val:.4f} → {last_val:.4f} ({delta:+.4f})")

        return "\n".join(lines)

    def get_signal_list(self) -> str:
        """List currently tracked signals."""
        if not self._state.signals:
            return "No signals are being tracked."
        lines = ["Currently tracked signals:"]
        for sid, sig in self._state.signals.items():
            n_pts = len(sig.history)
            unit_str = f" ({sig.unit})" if sig.unit else ""
            last_val = ""
            if n_pts > 0 and sig.history[-1].value is not None:
                last_val = f" = {sig.history[-1].value:.4f}"
            lines.append(f"  - {sig.display_name}{unit_str}{last_val}  [{n_pts} snapshots]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level functions for chat_tools integration
# ---------------------------------------------------------------------------

def run_signal_tracker(
    model: NeqSimProcessModel,
    tracker: SignalTracker,
    action: str = "snapshot",
    signals: Optional[List[str]] = None,
    label: str = "",
    signal_ref: str = "",
) -> str:
    """Execute a signal tracker action.

    Actions:
      - "add":       Add signals to track (signals = list of descriptions)
      - "remove":    Remove a tracked signal (signal_ref = identifier)
      - "snapshot":  Record current values of all tracked signals
      - "show":      Return the trend table
      - "list":      List currently tracked signals
      - "clear":     Remove all signals and history
    """
    if action == "add":
        if not signals:
            return "No signals specified to track."
        added = []
        for desc in signals:
            sid, display = tracker.add_signal(model, desc)
            added.append(display)
        # Take an initial snapshot
        tracker.snapshot(model, label=label or "Initial")
        return f"Now tracking: {', '.join(added)}\n\n{tracker.get_trend_table()}"

    elif action == "remove":
        ok = tracker.remove_signal(signal_ref)
        if ok:
            return f"Stopped tracking '{signal_ref}'.\n\n{tracker.get_signal_list()}"
        return f"Signal '{signal_ref}' not found in tracked signals.\n\n{tracker.get_signal_list()}"

    elif action == "snapshot":
        if not tracker.has_signals():
            return "No signals are being tracked. Use 'track <signal>' to start."
        tracker.snapshot(model, label=label)
        return tracker.get_trend_table()

    elif action == "show":
        return tracker.get_trend_table()

    elif action == "list":
        return tracker.get_signal_list()

    elif action == "clear":
        tracker.clear()
        return "All tracked signals cleared."

    return f"Unknown tracker action: {action}"


def format_signal_tracker_result(text: str) -> str:
    """Pass-through formatter (result is already text)."""
    return text
