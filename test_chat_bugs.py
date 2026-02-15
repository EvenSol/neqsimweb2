"""
Test script to identify bugs in the process chat functionality.
"""
import sys
sys.path.insert(0, '.')

from process_chat.chat_tools import extract_property_query, extract_scenario_json
from process_chat.process_model import NeqSimProcessModel
from process_chat.patch_schema import Scenario, InputPatch, scenarios_from_json
from process_chat.scenario_engine import run_scenarios

def test_extract_property_query():
    print("=== extract_property_query tests ===")

    # Test 1: Normal query block
    t1 = 'Here:\n```query\n{"properties": ["compressor sizing"]}\n```\nDone.'
    r1 = extract_property_query(t1)
    assert r1 is not None, "Test 1 FAILED: should find query"
    assert r1["properties"] == ["compressor sizing"], f"Test 1 FAILED: {r1}"
    print("  Test 1 PASS")

    # Test 2: No query block
    t2 = "No query here."
    r2 = extract_property_query(t2)
    assert r2 is None, "Test 2 FAILED: should return None"
    print("  Test 2 PASS")

    # Test 3: Multiple properties
    t3 = '```query\n{"properties": ["feed gas TVP", "separator gasLoadFactor"]}\n```'
    r3 = extract_property_query(t3)
    assert r3 is not None, "Test 3 FAILED"
    assert len(r3["properties"]) == 2, f"Test 3 FAILED: {r3}"
    print("  Test 3 PASS")

    # Test 4: Indented JSON inside query block
    t4 = '```query\n  {"properties": ["TVP"]}\n```'
    r4 = extract_property_query(t4)
    # The regex pattern requires newline-delimited, indentation may cause issues
    print(f"  Test 4 (indented): {'PASS' if r4 else 'FAIL - indented JSON not matched'}")

    # Test 5: Query block should not be picked up by json extractor
    t5 = '```query\n{"properties": ["TVP"]}\n```'
    r5_q = extract_property_query(t5)
    r5_j = extract_scenario_json(t5)
    assert r5_q is not None, "Test 5 FAILED: query not found"
    assert r5_j is None, "Test 5 FAILED: json should not match query block"
    print("  Test 5 PASS")


def test_extract_scenario_json():
    print("\n=== extract_scenario_json tests ===")

    # Test 1: Normal JSON block
    t1 = '```json\n{"scenarios": [{"name": "test", "description": "test", "patch": {"changes": {}}}]}\n```'
    r1 = extract_scenario_json(t1)
    assert r1 is not None, "Test 1 FAILED"
    assert "scenarios" in r1, "Test 1 FAILED: no scenarios key"
    print("  Test 1 PASS")

    # Test 2: No JSON block
    t2 = "The answer is 42."
    r2 = extract_scenario_json(t2)
    assert r2 is None, "Test 2 FAILED"
    print("  Test 2 PASS")

    # Test 3: JSON block without scenarios key (should not match)
    t3 = '```json\n{"data": 123}\n```'
    r3 = extract_scenario_json(t3)
    assert r3 is None, "Test 3 FAILED: should not match non-scenario JSON"
    print("  Test 3 PASS")


def test_model_basic():
    print("\n=== Model basic tests ===")
    model = NeqSimProcessModel.from_file('test_process.neqsim')

    # Test list_units returns valid data
    units = model.list_units()
    assert len(units) > 0, "No units found"
    print(f"  Units: {len(units)} OK")

    # Test list_streams returns valid data
    streams = model.list_streams()
    assert len(streams) > 0, "No streams found"
    print(f"  Streams: {len(streams)} OK")

    # Test get_unit with valid name
    u = model.get_unit('1st stage compressor')
    assert u is not None
    print("  get_unit OK")

    # Test get_unit with invalid name
    try:
        model.get_unit('nonexistent unit')
        print("  get_unit(nonexistent) FAIL: should raise KeyError")
    except KeyError:
        print("  get_unit(nonexistent) OK: raises KeyError")

    # Test get_stream with valid name
    s = model.get_stream('feed gas')
    assert s is not None
    print("  get_stream OK")

    # Test get_stream with invalid name
    try:
        model.get_stream('nonexistent stream')
        print("  get_stream(nonexistent) FAIL: should raise KeyError")
    except KeyError:
        print("  get_stream(nonexistent) OK: raises KeyError")

    return model


def test_run_and_results(model):
    print("\n=== Run & results tests ===")
    result = model.run()

    # Check KPIs exist
    assert 'total_power_kW' in result.kpis, "Missing total_power_kW"
    assert 'total_duty_kW' in result.kpis, "Missing total_duty_kW"
    assert result.kpis['total_power_kW'].value > 0, "total_power_kW should be > 0"
    print(f"  total_power_kW = {result.kpis['total_power_kW'].value:.2f} OK")

    # Check mass balance constraint
    assert len(result.constraints) > 0, "No constraints"
    mb = [c for c in result.constraints if c.name == 'mass_balance']
    assert len(mb) == 1, "Missing mass_balance constraint"
    assert mb[0].status == 'OK', f"Mass balance not OK: {mb[0].detail}"
    print(f"  Mass balance: {mb[0].status} OK")

    # Check no convergence warning
    cw = [c for c in result.constraints if c.name == 'convergence']
    if cw:
        print(f"  Convergence warning: {cw[0].detail}")
    else:
        print("  No convergence warning OK")

    # Check equipment-specific KPIs exist
    compressor_kpis = [k for k in result.kpis if 'compressor' in k.lower() and 'compressionRatio' in k]
    assert len(compressor_kpis) > 0, "Missing compressor compressionRatio KPIs"
    print(f"  Equipment KPIs: {len(compressor_kpis)} compressionRatio entries OK")

    # Check sizing KPIs exist
    sizing_kpis = [k for k in result.kpis if '.sizing.' in k]
    assert len(sizing_kpis) > 0, "Missing sizing KPIs"
    print(f"  Sizing KPIs: {len(sizing_kpis)} entries OK")

    # Check mechanical design KPIs exist
    mech_kpis = [k for k in result.kpis if '.mechDesign.' in k]
    assert len(mech_kpis) > 0, "Missing mechanical design KPIs"
    print(f"  Mechanical Design KPIs: {len(mech_kpis)} entries OK")

    # Check cost KPIs exist
    cost_kpis = [k for k in result.kpis if '.cost.' in k]
    assert len(cost_kpis) > 0, "Missing cost KPIs"
    print(f"  Cost KPIs: {len(cost_kpis)} entries OK")

    # Check system-level KPIs exist
    system_kpis = [k for k in result.kpis if k.startswith('system.')]
    assert len(system_kpis) > 0, "Missing system-level KPIs"
    print(f"  System KPIs: {len(system_kpis)} entries OK")

    # Check specific system KPIs
    assert 'system.totalWeight_kg' in result.kpis, "Missing system.totalWeight_kg"
    assert 'system.plotSpace_m2' in result.kpis, "Missing system.plotSpace_m2"
    assert 'system.totalCost_USD' in result.kpis, "Missing system.totalCost_USD"
    assert result.kpis['system.totalWeight_kg'].value > 0, "system total weight should be > 0"
    assert result.kpis['system.plotSpace_m2'].value > 0, "system plot space should be > 0"
    print(f"  System total weight: {result.kpis['system.totalWeight_kg'].value:.0f} kg OK")
    print(f"  System plot space: {result.kpis['system.plotSpace_m2'].value:.1f} m2 OK")
    print(f"  System total cost: {result.kpis['system.totalCost_USD'].value:.0f} USD OK")

    return result


def test_query_properties(model):
    print("\n=== query_properties tests ===")

    # Test equipment queries
    r1 = model.query_properties("compressor compressionRatio")
    assert "compressionRatio" in r1, f"Should find compressionRatio: {r1}"
    print("  compressor compressionRatio OK")

    r2 = model.query_properties("separator gasLoadFactor")
    assert "gasLoadFactor" in r2, f"Should find gasLoadFactor: {r2}"
    print("  separator gasLoadFactor OK")

    r3 = model.query_properties("inlet separator sizing")
    assert "sizing" in r3 and "inlet separator" in r3, f"Should find inlet separator sizing: {r3}"
    print("  inlet separator sizing OK")

    r4 = model.query_properties("compressor entropy")
    assert "entropy" in r4, f"Should find entropy: {r4}"
    print("  compressor entropy OK")

    # Test stream property queries
    r5 = model.query_properties("feed gas TVP")
    assert "TVP" in r5, f"Should find TVP: {r5}"
    print("  feed gas TVP OK")

    # Test nonexistent
    r6 = model.query_properties("widget fluxcapacitor")
    assert "No properties matching" in r6, f"Should say no match: {r6}"
    print("  nonexistent property OK")

    # Test mechanical design queries
    r7 = model.query_properties("separator mechDesign wallThickness")
    assert "wallThickness" in r7, f"Should find wallThickness: {r7}"
    print("  separator mechDesign wallThickness OK")

    r8 = model.query_properties("compressor mechDesign weightTotal")
    assert "weightTotal" in r8, f"Should find weightTotal: {r8}"
    print("  compressor mechDesign weightTotal OK")

    r9 = model.query_properties("system totalWeight")
    assert "totalWeight" in r9, f"Should find system totalWeight: {r9}"
    print("  system totalWeight OK")

    r10 = model.query_properties("system plotSpace")
    assert "plotSpace" in r10, f"Should find system plotSpace: {r10}"
    print("  system plotSpace OK")

    r11 = model.query_properties("system totalCost")
    assert "totalCost" in r11, f"Should find system totalCost: {r11}"
    print("  system totalCost OK")

    r12 = model.query_properties("cost totalCost")
    assert "totalCost" in r12, f"Should find cost totalCost: {r12}"
    print("  cost totalCost OK")

    r13 = model.query_properties("mechDesign module")
    assert "module" in r13.lower(), f"Should find module dimensions: {r13}"
    print("  mechDesign module dimensions OK")

    r14 = model.query_properties("system weightByType")
    assert "weightByType" in r14, f"Should find weightByType: {r14}"
    print("  system weightByType OK")


def test_scenario_execution(model):
    print("\n=== Scenario execution tests ===")

    # Test 1: Simple parameter change
    sc = Scenario(
        name='Test pressure change',
        description='Test',
        patch=InputPatch(changes={
            'units.2nd stage compressor.outletpressure_bara': 150.0
        })
    )
    comparison = run_scenarios(model, [sc])
    assert comparison.base.success, "Base should succeed"
    assert len(comparison.cases) == 1, "Should have 1 case"
    assert comparison.cases[0].success, f"Case should succeed: {comparison.cases[0].error}"
    base_power = comparison.base.result.kpis['total_power_kW'].value
    case_power = comparison.cases[0].result.kpis['total_power_kW'].value
    assert case_power > base_power, f"Higher pressure should mean more power: {base_power} -> {case_power}"
    print(f"  Pressure change: power {base_power:.1f} -> {case_power:.1f} OK")

    # Test 2: Invalid unit name
    sc2 = Scenario(
        name='Invalid unit',
        description='Test',
        patch=InputPatch(changes={
            'units.nonexistent.outletpressure_bara': 100.0
        })
    )
    comparison2 = run_scenarios(model, [sc2])
    # The scenario should still run the base OK
    assert comparison2.base.success, "Base should still succeed"
    # The case patch should fail but scenario might still run
    print(f"  Invalid unit test: case success={comparison2.cases[0].success}")

    # Test 3: Relative change (scale)
    sc3 = Scenario(
        name='Scale flow',
        description='Test',
        patch=InputPatch(changes={
            'streams.feed gas.flow_kg_hr': {"op": "scale", "value": 1.1}
        })
    )
    comparison3 = run_scenarios(model, [sc3])
    assert comparison3.cases[0].success, f"Scale should succeed: {comparison3.cases[0].error}"
    print("  Scale flow OK")


def test_chat_flow():
    """Test the full chat-tools flow without calling Gemini."""
    print("\n=== Chat flow structure tests ===")

    from process_chat.chat_tools import build_system_prompt, format_comparison_for_llm

    model = NeqSimProcessModel.from_file('test_process.neqsim')

    # Test system prompt builds without errors
    prompt = build_system_prompt(model)
    assert len(prompt) > 100, "System prompt too short"
    assert "PROPERTY QUERY" in prompt, "Missing PROPERTY QUERY section"
    assert "equipment_type" in prompt, "Missing equipment_type instructions"
    assert "gasLoadFactor" in prompt, "Missing gasLoadFactor in prompt"
    assert "compressionRatio" in prompt, "Missing compressionRatio in prompt"
    assert "mechDesign" in prompt, "Missing mechDesign in prompt"
    assert "wallThickness" in prompt, "Missing wallThickness in prompt"
    assert "totalWeight_kg" in prompt, "Missing totalWeight_kg in prompt"
    assert "totalCost_USD" in prompt, "Missing totalCost_USD in prompt"
    assert "plotSpace_m2" in prompt, "Missing plotSpace_m2 in prompt"
    print(f"  System prompt: {len(prompt)} chars OK")

    # Test format_comparison_for_llm
    sc = Scenario(name='Test', description='Test',
                  patch=InputPatch(changes={'units.2nd stage compressor.outletpressure_bara': 150.0}))
    comparison = run_scenarios(model, [sc])
    text = format_comparison_for_llm(comparison)
    assert len(text) > 0, "Empty comparison text"
    print(f"  Comparison text: {len(text)} chars OK")


def test_model_summary():
    """Test get_model_summary for completeness."""
    print("\n=== Model summary tests ===")
    model = NeqSimProcessModel.from_file('test_process.neqsim')
    summary = model.get_model_summary()

    assert "Process Model Summary" in summary
    assert "Process Topology" in summary
    assert "All Streams" in summary
    assert "Compressor" in summary
    assert "Cooler" in summary
    assert "Separator" in summary
    # Check that properties are shown
    assert "power_kW" in summary
    assert "duty_kW" in summary
    # Check stream conditions
    assert "T=" in summary
    assert "P=" in summary
    print(f"  Summary: {len(summary)} chars, all checks PASS")


if __name__ == "__main__":
    test_extract_property_query()
    test_extract_scenario_json()
    model = test_model_basic()
    test_run_and_results(model)
    test_query_properties(model)
    test_scenario_execution(model)
    test_chat_flow()
    test_model_summary()
    print("\nAll tests passed!")
