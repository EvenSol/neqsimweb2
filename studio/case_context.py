"""Shared, UI-independent lifecycle state for NeqSim Studio cases.

The existing Process Flowsheet Studio case dictionary remains the authoritative
portable case format.  This module adds only session-level workspace metadata so
multiple Studio pages can agree on the active case without changing or wrapping
Classic-compatible JSON files.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, MutableMapping
from uuid import uuid4


CONTEXT_SCHEMA_VERSION = 1
MAX_CASE_FILE_BYTES = 1_000_000
MAX_RECENT_CASES = 8

STUDIO_CASE_CONTEXT_STATE_KEY = "neqsim_studio_case_context"
STUDIO_PENDING_CASE_STATE_KEY = "neqsim_studio_pending_case"
STUDIO_RECENT_CASES_STATE_KEY = "neqsim_studio_recent_cases"

STATUS_DRAFT = "draft"
STATUS_DIRTY = "dirty"
STATUS_SOLVING = "solving"
STATUS_SOLVED = "solved"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"
STATUS_TIMED_OUT = "timed-out"
STATUS_INVALID = "invalid"
VALID_STATUSES = {
    STATUS_DRAFT,
    STATUS_DIRTY,
    STATUS_SOLVING,
    STATUS_SOLVED,
    STATUS_WARNING,
    STATUS_FAILED,
    STATUS_TIMED_OUT,
    STATUS_INVALID,
}

PENDING_NEW = "new"
PENDING_OPEN = "open"


def _utc_now() -> str:
    """Return a stable UTC timestamp suitable for JSON metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, *, fallback: str = "") -> str:
    return str(value).strip() if value is not None else fallback


def _clone_json(value: Any) -> Any:
    """Deep-copy one JSON-compatible value and reject non-finite data."""

    return json.loads(json.dumps(value, allow_nan=False))


def case_fingerprint(case_spec: Mapping[str, Any]) -> str:
    """Return a deterministic identity for one portable case specification."""

    encoded = json.dumps(
        case_spec,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_portable_case(case_spec: Any) -> dict[str, Any]:
    """Validate the format-neutral envelope without duplicating page validation.

    Detailed schema migration, topology, units, and engineering validation stay
    with the existing Process Flowsheet Studio importer.  This boundary prevents
    the shared workspace from inventing a second case schema.
    """

    if not isinstance(case_spec, dict):
        raise ValueError("The Studio case root must be a JSON object.")
    name = _clean_text(case_spec.get("name"))
    if not name:
        raise ValueError("The Studio case must have a non-empty name.")
    if len(name) > 120:
        raise ValueError("The Studio case name cannot exceed 120 characters.")
    schema_version = case_spec.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        raise ValueError("The Studio case requires a positive integer schema_version.")
    if not isinstance(case_spec.get("fluid"), dict):
        raise ValueError("The Studio case must contain a fluid object.")
    if not isinstance(case_spec.get("process"), list):
        raise ValueError("The Studio case must contain a process array.")
    return _clone_json(case_spec)


def decode_portable_case(
    payload: bytes,
    *,
    max_bytes: int = MAX_CASE_FILE_BYTES,
) -> dict[str, Any]:
    """Decode one uploaded Classic-compatible case without mutating session state."""

    if not isinstance(payload, bytes):
        raise ValueError("The uploaded Studio case must be bytes.")
    if len(payload) > max_bytes:
        raise ValueError(f"The Studio case file cannot exceed {max_bytes} bytes.")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The Studio case must use UTF-8 encoding.") from exc
    try:
        case_spec = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The Studio case is not valid JSON: {exc.msg}.") from exc
    return validate_portable_case(case_spec)


def encode_portable_case(case_spec: Mapping[str, Any]) -> bytes:
    """Serialize the unchanged portable case contract for download."""

    validated = validate_portable_case(dict(case_spec))
    return (json.dumps(validated, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _thermodynamic_summary(case_spec: Mapping[str, Any]) -> dict[str, Any]:
    fluid = case_spec.get("fluid", {})
    return {
        "eos_model": _clean_text(fluid.get("eos_model"), fallback="unknown").upper(),
        "mixing_rule": fluid.get("mixing_rule"),
        "composition_basis": _clean_text(fluid.get("composition_basis")),
    }


def _unit_summary(case_spec: Mapping[str, Any]) -> dict[str, str]:
    fluid = case_spec.get("fluid", {})
    return {
        "system": "SI",
        "temperature": "degC",
        "pressure": "bara",
        "flow": _clean_text(fluid.get("flow_unit"), fallback="kg/hr"),
    }


def build_case_context(
    case_spec: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    status: str = STATUS_DIRTY,
    error: str | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
    solved_signature: str | None = None,
    model_available: bool = False,
    model_name: str | None = None,
    source: str = "Process Flowsheet Studio",
    now: str | None = None,
) -> dict[str, Any]:
    """Create or update the shared active-case context.

    Identity and creation time survive edits. Modified time changes only when a
    portable case or observable lifecycle field changes.
    """

    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported Studio case status: {status}")
    validated = validate_portable_case(dict(case_spec))
    fingerprint = case_fingerprint(validated)
    timestamp = now or _utc_now()
    previous_context = dict(previous) if isinstance(previous, Mapping) else {}
    previous_provenance = previous_context.get("provenance", {})
    if not isinstance(previous_provenance, Mapping):
        previous_provenance = {}

    clean_warnings = [
        _clean_text(item)
        for item in (warnings or [])
        if _clean_text(item)
    ]
    runtime = {
        "model_available": bool(model_available),
        "model_name": _clean_text(model_name) or None,
        "solved_signature": _clean_text(solved_signature) or None,
    }
    observable = {
        "case_fingerprint": fingerprint,
        "status": status,
        "error": _clean_text(error) or None,
        "warnings": clean_warnings,
        "runtime": runtime,
    }
    previous_observable = {
        key: previous_context.get(key) for key in observable
    }
    modified_at = previous_provenance.get("modified_at", timestamp)
    if observable != previous_observable:
        modified_at = timestamp

    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "case_id": _clean_text(previous_context.get("case_id")) or str(uuid4()),
        "name": validated["name"],
        "description": _clean_text(validated.get("description")),
        "status": status,
        "dirty": status not in {STATUS_SOLVED, STATUS_WARNING},
        "error": observable["error"],
        "warnings": clean_warnings,
        "units": _unit_summary(validated),
        "thermodynamics": _thermodynamic_summary(validated),
        "case_schema_version": validated["schema_version"],
        "case_fingerprint": fingerprint,
        "case_spec": validated,
        "runtime": runtime,
        "provenance": {
            "source": _clean_text(source, fallback="Process Flowsheet Studio"),
            "created_at": previous_provenance.get("created_at", timestamp),
            "modified_at": modified_at,
        },
    }


def get_active_case(session_state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a defensive copy of the active context, if present."""

    context = session_state.get(STUDIO_CASE_CONTEXT_STATE_KEY)
    return deepcopy(context) if isinstance(context, dict) else None


def set_active_case(
    session_state: MutableMapping[str, Any],
    case_spec: Mapping[str, Any],
    **lifecycle: Any,
) -> dict[str, Any]:
    """Synchronize the active context and retain it in recent cases."""

    context = build_case_context(
        case_spec,
        previous=session_state.get(STUDIO_CASE_CONTEXT_STATE_KEY),
        **lifecycle,
    )
    session_state[STUDIO_CASE_CONTEXT_STATE_KEY] = context
    _record_recent_case(session_state, context)
    return deepcopy(context)


def clear_active_case(
    session_state: MutableMapping[str, Any],
    *,
    clear_recent: bool = False,
) -> None:
    """Clear only Studio-owned lifecycle state."""

    session_state.pop(STUDIO_CASE_CONTEXT_STATE_KEY, None)
    session_state.pop(STUDIO_PENDING_CASE_STATE_KEY, None)
    if clear_recent:
        session_state.pop(STUDIO_RECENT_CASES_STATE_KEY, None)


def save_case_as(
    session_state: MutableMapping[str, Any],
    name: str,
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Clone the active portable case under a new identity and name."""

    context = get_active_case(session_state)
    if context is None:
        raise ValueError("There is no active Studio case to save as.")
    clean_name = _clean_text(name)
    if not clean_name:
        raise ValueError("The new Studio case name cannot be empty.")
    if len(clean_name) > 120:
        raise ValueError("The Studio case name cannot exceed 120 characters.")
    case_spec = _clone_json(context["case_spec"])
    case_spec["name"] = clean_name
    cloned = build_case_context(
        case_spec,
        status=STATUS_DIRTY,
        source=f"Save As from {context['case_id']}",
        now=now,
    )
    session_state[STUDIO_CASE_CONTEXT_STATE_KEY] = cloned
    _record_recent_case(session_state, cloned)
    return deepcopy(cloned)


def queue_new_case(session_state: MutableMapping[str, Any]) -> None:
    """Request the existing flowsheet page to initialize its validated template."""

    session_state[STUDIO_PENDING_CASE_STATE_KEY] = {"action": PENDING_NEW}


def queue_open_case(
    session_state: MutableMapping[str, Any],
    case_spec: Mapping[str, Any],
    *,
    preserve_identity: bool = False,
) -> None:
    """Request detailed import/migration by the existing flowsheet page."""

    session_state[STUDIO_PENDING_CASE_STATE_KEY] = {
        "action": PENDING_OPEN,
        "case_spec": validate_portable_case(dict(case_spec)),
        "preserve_identity": bool(preserve_identity),
    }


def queue_recent_case(session_state: MutableMapping[str, Any], case_id: str) -> None:
    """Queue one recent case for detailed import by the flowsheet page."""

    clean_id = _clean_text(case_id)
    for context in recent_cases(session_state):
        if context.get("case_id") == clean_id:
            session_state[STUDIO_CASE_CONTEXT_STATE_KEY] = deepcopy(context)
            queue_open_case(
                session_state,
                context["case_spec"],
                preserve_identity=True,
            )
            return
    raise ValueError("The selected recent Studio case is no longer available.")


def consume_pending_case(
    session_state: MutableMapping[str, Any],
) -> dict[str, Any] | None:
    """Consume a one-shot cross-page lifecycle request."""

    pending = session_state.pop(STUDIO_PENDING_CASE_STATE_KEY, None)
    if not isinstance(pending, dict):
        return None
    action = pending.get("action")
    if action == PENDING_NEW:
        return {"action": PENDING_NEW}
    if action == PENDING_OPEN:
        return {
            "action": PENDING_OPEN,
            "case_spec": validate_portable_case(pending.get("case_spec")),
            "preserve_identity": bool(pending.get("preserve_identity")),
        }
    raise ValueError("Unsupported pending Studio case action.")


def recent_cases(session_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return newest-first defensive copies of bounded session-local cases."""

    recent = session_state.get(STUDIO_RECENT_CASES_STATE_KEY)
    if not isinstance(recent, list):
        return []
    return [deepcopy(item) for item in reversed(recent) if isinstance(item, dict)]


def _record_recent_case(
    session_state: MutableMapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    recent = session_state.get(STUDIO_RECENT_CASES_STATE_KEY)
    recent_items = list(recent) if isinstance(recent, list) else []
    case_id = context.get("case_id")
    cleaned = [
        deepcopy(item)
        for item in recent_items
        if isinstance(item, dict) and item.get("case_id") != case_id
    ]
    cleaned.append(deepcopy(dict(context)))
    session_state[STUDIO_RECENT_CASES_STATE_KEY] = cleaned[-MAX_RECENT_CASES:]
