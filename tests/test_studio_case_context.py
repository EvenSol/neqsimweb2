"""Unit tests for the shared NeqSim Studio case lifecycle."""

from __future__ import annotations

import json
import unittest

from studio.case_context import (
    MAX_RECENT_CASES,
    PENDING_NEW,
    PENDING_OPEN,
    STATUS_DIRTY,
    STATUS_FAILED,
    STATUS_SOLVED,
    STUDIO_CASE_CONTEXT_STATE_KEY,
    build_case_context,
    case_fingerprint,
    clear_active_case,
    consume_pending_case,
    decode_portable_case,
    encode_portable_case,
    get_active_case,
    queue_new_case,
    queue_open_case,
    queue_recent_case,
    recent_cases,
    save_case_as,
    set_active_case,
)


def sample_case(name: str = "Compression Case", pressure: float = 70.0) -> dict:
    return {
        "schema_version": 4,
        "name": name,
        "description": "A portable Process Flowsheet Studio case.",
        "fluid": {
            "eos_model": "srk",
            "mixing_rule": 2,
            "composition_basis": "mole_fraction",
            "flow_unit": "kg/hr",
            "pressure_bara": pressure,
        },
        "process": [{"name": "feed gas", "type": "stream"}],
        "units": [],
        "connections": [],
        "subflowsheets": [],
    }


class StudioCaseContextTest(unittest.TestCase):
    def test_context_keeps_portable_v4_case_authoritative(self):
        case_spec = sample_case()
        context = build_case_context(
            case_spec,
            status=STATUS_DIRTY,
            now="2026-08-10T00:00:00+00:00",
        )

        self.assertEqual(context["case_spec"], case_spec)
        self.assertEqual(context["case_schema_version"], 4)
        self.assertEqual(context["thermodynamics"]["eos_model"], "SRK")
        self.assertEqual(context["units"]["flow"], "kg/hr")
        self.assertTrue(context["dirty"])

    def test_repeated_sync_preserves_identity_and_modified_time(self):
        initial = build_case_context(
            sample_case(),
            now="2026-08-10T00:00:00+00:00",
        )
        repeated = build_case_context(
            sample_case(),
            previous=initial,
            now="2026-08-10T01:00:00+00:00",
        )

        self.assertEqual(repeated["case_id"], initial["case_id"])
        self.assertEqual(
            repeated["provenance"]["modified_at"],
            initial["provenance"]["modified_at"],
        )

    def test_case_edit_changes_fingerprint_and_modified_time(self):
        initial = build_case_context(
            sample_case(pressure=70.0),
            now="2026-08-10T00:00:00+00:00",
        )
        edited = build_case_context(
            sample_case(pressure=75.0),
            previous=initial,
            now="2026-08-10T01:00:00+00:00",
        )

        self.assertNotEqual(edited["case_fingerprint"], initial["case_fingerprint"])
        self.assertEqual(
            edited["provenance"]["modified_at"],
            "2026-08-10T01:00:00+00:00",
        )

    def test_solved_runtime_evidence_is_explicit(self):
        context = build_case_context(
            sample_case(),
            status=STATUS_SOLVED,
            solved_signature="abc123",
            model_available=True,
            model_name="process_flowsheet_studio.neqsim",
        )

        self.assertFalse(context["dirty"])
        self.assertEqual(context["runtime"]["solved_signature"], "abc123")
        self.assertTrue(context["runtime"]["model_available"])

    def test_failed_state_keeps_actionable_error(self):
        context = build_case_context(
            sample_case(),
            status=STATUS_FAILED,
            error="Native solve failed",
        )

        self.assertTrue(context["dirty"])
        self.assertEqual(context["error"], "Native solve failed")

    def test_raw_case_round_trip_does_not_add_context_wrapper(self):
        encoded = encode_portable_case(sample_case())
        decoded = decode_portable_case(encoded)

        self.assertEqual(decoded, sample_case())
        self.assertNotIn("context_schema_version", json.loads(encoded))

    def test_invalid_upload_does_not_mutate_existing_context(self):
        state = {}
        set_active_case(state, sample_case())
        before = get_active_case(state)

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            decode_portable_case(b"{broken")

        self.assertEqual(get_active_case(state), before)

    def test_save_as_creates_new_identity_and_renamed_portable_case(self):
        state = {}
        original = set_active_case(state, sample_case())
        cloned = save_case_as(
            state,
            "Compression Case B",
            now="2026-08-10T01:00:00+00:00",
        )

        self.assertNotEqual(cloned["case_id"], original["case_id"])
        self.assertEqual(cloned["name"], "Compression Case B")
        self.assertEqual(cloned["case_spec"]["name"], "Compression Case B")
        self.assertEqual(cloned["status"], STATUS_DIRTY)

    def test_pending_actions_are_one_shot(self):
        state = {}
        queue_new_case(state)
        self.assertEqual(consume_pending_case(state), {"action": PENDING_NEW})
        self.assertIsNone(consume_pending_case(state))

        queue_open_case(state, sample_case())
        pending = consume_pending_case(state)
        self.assertEqual(pending["action"], PENDING_OPEN)
        self.assertEqual(pending["case_spec"], sample_case())
        self.assertFalse(pending["preserve_identity"])
        self.assertIsNone(consume_pending_case(state))

    def test_recent_cases_are_bounded_and_reopenable(self):
        state = {}
        for index in range(MAX_RECENT_CASES + 2):
            set_active_case(state, sample_case(name=f"Case {index}"))
            save_case_as(state, f"Saved {index}")

        recent = recent_cases(state)
        self.assertEqual(len(recent), MAX_RECENT_CASES)
        selected = recent[0]
        queue_recent_case(state, selected["case_id"])
        pending = consume_pending_case(state)
        self.assertEqual(pending["case_spec"]["name"], selected["name"])
        self.assertEqual(get_active_case(state)["case_id"], selected["case_id"])

    def test_opening_uploaded_case_starts_a_new_identity(self):
        state = {}
        original = set_active_case(state, sample_case("Original"))
        queue_open_case(state, sample_case("Uploaded"))

        self.assertEqual(get_active_case(state)["case_id"], original["case_id"])
        pending = consume_pending_case(state)
        clear_active_case(state)
        opened = set_active_case(state, pending["case_spec"])
        self.assertNotEqual(opened["case_id"], original["case_id"])

    def test_clear_does_not_touch_classic_session_keys(self):
        state = {"classic_case": {"keep": True}}
        set_active_case(state, sample_case())
        clear_active_case(state)

        self.assertNotIn(STUDIO_CASE_CONTEXT_STATE_KEY, state)
        self.assertEqual(state["classic_case"], {"keep": True})

    def test_fingerprint_is_order_independent(self):
        first = sample_case()
        second = dict(reversed(list(first.items())))
        self.assertEqual(case_fingerprint(first), case_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
