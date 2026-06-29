"""Test that structural additions from scenarios persist to the base model."""
import sys
sys.path.insert(0, '.')

from process_chat.process_builder import ProcessBuilder
from process_chat.chat_tools import ProcessChatSession, extract_scenario_json, extract_build_spec

# 1. Build a simple process (single feed stream)
print("=== Step 1: Build process ===")
spec = {
    'name': 'Methane Feed',
    'fluid': {
        'eos_model': 'srk',
        'components': {'methane': 1.0},
        'composition_basis': 'mole_fraction',
        'temperature_C': 25.0,
        'pressure_bara': 50.0,
        'total_flow': 100,
        'flow_unit': 'kg/hr',
    },
    'process': [
        {'name': 'methane feed', 'type': 'stream'},
    ],
}

builder = ProcessBuilder()
model = builder.build_from_spec(spec)
units = model.list_units()
print(f"  Initial units: {[u.name for u in units]}")
assert len(units) == 1, f"Expected 1 unit, got {len(units)}"
print("  PASSED")

# 2. Simulate what happens when a scenario with add_units is run
print()
print("=== Step 2: Apply scenario with add_units ===")
from process_chat.patch_schema import InputPatch, Scenario, AddUnitOp, scenarios_from_json
from process_chat.scenario_engine import run_scenarios, apply_add_units
from process_chat.process_model import NeqSimProcessModel

# Create a scenario that adds a cooler
scenario_json = {
    "scenarios": [{
        "name": "Add cooler",
        "description": "Add a cooler after methane feed",
        "patch": {
            "changes": {},
            "add_units": [{
                "name": "methane cooler",
                "equipment_type": "cooler",
                "insert_after": "methane feed",
                "params": {"outlet_temperature_C": 0.0}
            }]
        }
    }]
}

scenarios = scenarios_from_json(scenario_json)
comparison = run_scenarios(model, scenarios)

# Check that the scenario succeeded
assert comparison.cases[0].success, f"Scenario failed: {comparison.cases[0].error}"
print("  Scenario ran successfully on clone")

# Before fix: base model still has just 1 unit
units_before = model.list_units()
print(f"  Base model units BEFORE persistence: {[u.name for u in units_before]}")

# 3. Now simulate the persistence logic (what _handle_scenario does)
print()
print("=== Step 3: Persist structural additions to base model ===")
for sc in scenarios:
    has_structural = sc.patch.add_units or sc.patch.add_streams or sc.patch.add_process
    if not has_structural:
        continue
    case_ok = any(c.success and c.scenario.name == sc.name for c in comparison.cases)
    if not case_ok:
        continue
    if sc.patch.add_units:
        apply_add_units(model, sc.patch.add_units)
    NeqSimProcessModel._run_until_converged(model.get_process())
    model._index_model_objects()
    model.refresh_source_bytes()

units_after = model.list_units()
print(f"  Base model units AFTER persistence: {[u.name for u in units_after]}")
assert len(units_after) >= 2, f"Expected at least 2 units, got {len(units_after)}"
has_cooler = any("cooler" in u.name.lower() for u in units_after)
assert has_cooler, "Cooler not found in base model after persistence"
print("  Cooler persisted to base model!")
print("  PASSED")

# 4. Now verify we can add a separator after the cooler
print()
print("=== Step 4: Add separator after cooler ===")
scenario_json2 = {
    "scenarios": [{
        "name": "Add separator",
        "description": "Add separator after cooler",
        "patch": {
            "changes": {},
            "add_units": [{
                "name": "separator",
                "equipment_type": "separator",
                "insert_after": "methane cooler",
                "params": {}
            }]
        }
    }]
}

scenarios2 = scenarios_from_json(scenario_json2)
comparison2 = run_scenarios(model, scenarios2)

assert comparison2.cases[0].success, f"Scenario failed: {comparison2.cases[0].error}"
print("  Successfully added separator after cooler")

# Persist
for sc in scenarios2:
    if sc.patch.add_units:
        apply_add_units(model, sc.patch.add_units)
    NeqSimProcessModel._run_until_converged(model.get_process())
    model._index_model_objects()
    model.refresh_source_bytes()

units_final = model.list_units()
print(f"  Final units: {[u.name for u in units_final]}")
assert len(units_final) >= 3, f"Expected at least 3 units, got {len(units_final)}"
print("  PASSED")

# 5. Add a compressor after the separator
print()
print("=== Step 5: Add compressor after separator ===")
scenario_json3 = {
    "scenarios": [{
        "name": "Add compressor",
        "description": "Add compressor after separator gas outlet",
        "patch": {
            "changes": {},
            "add_units": [{
                "name": "gas compressor",
                "equipment_type": "compressor",
                "insert_after": "separator",
                "params": {"outlet_pressure_bara": 100.0, "isentropic_efficiency": 0.75}
            }]
        }
    }]
}

scenarios3 = scenarios_from_json(scenario_json3)
comparison3 = run_scenarios(model, scenarios3)

assert comparison3.cases[0].success, f"Scenario failed: {comparison3.cases[0].error}"
print("  Successfully added compressor after separator")

# Persist
for sc in scenarios3:
    if sc.patch.add_units:
        apply_add_units(model, sc.patch.add_units)
    NeqSimProcessModel._run_until_converged(model.get_process())
    model._index_model_objects()
    model.refresh_source_bytes()

units_final2 = model.list_units()
print(f"  Final units: {[u.name for u in units_final2]}")
assert len(units_final2) >= 4, f"Expected at least 4 units, got {len(units_final2)}"
print("  PASSED")

print()
print("=" * 50)
print("ALL PERSISTENCE TESTS PASSED")
print("=" * 50)
