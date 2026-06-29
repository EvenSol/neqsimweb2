"""End-to-end tests for ProcessBuilder and chat integration."""
import sys
sys.path.insert(0, '.')

# Test 1: ProcessBuilder build_from_spec
print('=== Test 1: Build from spec ===')
from process_chat.process_builder import ProcessBuilder

spec = {
    'name': 'Test Gas Compression',
    'fluid': {
        'eos_model': 'srk',
        'components': {'methane': 0.85, 'ethane': 0.07, 'propane': 0.03, 'CO2': 0.02, 'nitrogen': 0.03},
        'composition_basis': 'mole_fraction',
        'temperature_C': 25.0,
        'pressure_bara': 50.0,
        'total_flow': 10000,
        'flow_unit': 'kg/hr',
    },
    'process': [
        {'name': 'feed gas', 'type': 'stream'},
        {'name': 'inlet separator', 'type': 'separator'},
        {'name': '1st stage compressor', 'type': 'compressor',
         'params': {'outlet_pressure_bara': 100.0, 'isentropic_efficiency': 0.75}},
        {'name': 'intercooler', 'type': 'cooler',
         'params': {'outlet_temperature_C': 35.0}},
        {'name': 'export scrubber', 'type': 'gas_scrubber'},
    ],
}

builder = ProcessBuilder()
model = builder.build_from_spec(spec)
print(f'  Model: {model}')
units = model.list_units()
streams = model.list_streams()
print(f'  Units: {len(units)}')
print(f'  Streams: {len(streams)}')
for entry in builder.build_log:
    print(f'    - {entry}')
assert len(units) >= 4, f"Expected at least 4 units, got {len(units)}"
assert len(streams) >= 2, f"Expected at least 2 streams, got {len(streams)}"
print('  PASSED')

# Test 2: Python script generation
print()
print('=== Test 2: Python script ===')
script = builder.to_python_script()
assert len(script) > 200, f"Script too short: {len(script)}"
assert 'from neqsim import jneqsim' in script
assert 'SystemSrkEos' in script
assert "addComponent('methane'" in script
assert 'process.run()' in script
assert 'save_neqsim' in script
print(f'  Script length: {len(script)} chars')
print('  PASSED')

# Test 3: .neqsim file save
print()
print('=== Test 3: Save .neqsim bytes ===')
raw = builder.save_neqsim_bytes()
assert raw is not None, "save_neqsim_bytes returned None"
assert len(raw) > 1000, f".neqsim file too small: {len(raw)} bytes"
print(f'  Bytes: {len(raw)} bytes')
print('  PASSED')

# Test 4: Build summary
print()
print('=== Test 4: Build summary ===')
summary = builder.get_build_summary()
assert 'Test Gas Compression' in summary
assert 'methane' in summary
print(summary)
print('  PASSED')

# Test 5: extract_build_spec
print()
print('=== Test 5: extract_build_spec ===')
from process_chat.chat_tools import extract_build_spec

test_text = 'Here is the process:\n```build\n{"name": "test", "fluid": {}, "process": []}\n```'
result = extract_build_spec(test_text)
assert result is not None, "Failed to extract build spec"
assert result['name'] == 'test'
print(f'  Extracted: {result}')
print('  PASSED')

# Test 6: extract show_script action
print()
print('=== Test 6: extract show_script ===')
test_text2 = '```build\n{"action": "show_script"}\n```'
result2 = extract_build_spec(test_text2)
assert result2 is not None
assert result2['action'] == 'show_script'
print(f'  Show script: {result2}')
print('  PASSED')

# Test 7: extract save action
print()
print('=== Test 7: extract save ===')
test_text3 = '```build\n{"action": "save"}\n```'
result3 = extract_build_spec(test_text3)
assert result3 is not None
assert result3['action'] == 'save'
print(f'  Save action: {result3}')
print('  PASSED')

# Test 8: ProcessChatSession with model=None
print()
print('=== Test 8: ProcessChatSession builder mode ===')
from process_chat.chat_tools import ProcessChatSession, build_builder_system_prompt

# Just test construction — no API key needed for that
session = ProcessChatSession(model=None, api_key="test_key")
assert session.model is None
assert session._builder is None
assert 'BUILD' in session._system_prompt or 'build' in session._system_prompt.lower()
print('  Session created in builder mode')
print('  PASSED')

# Test 9: ProcessChatSession with model
print()
print('=== Test 9: ProcessChatSession model mode ===')
session2 = ProcessChatSession(model=model, api_key="test_key")
assert session2.model is model
assert 'equipment' in session2._system_prompt.lower() or 'process' in session2._system_prompt.lower()
print('  Session created in model mode')
print('  PASSED')

# Test 10: Builder with liquid outlet
print()
print('=== Test 10: Liquid outlet chaining ===')
spec2 = {
    'name': 'Liquid Processing',
    'fluid': {
        'eos_model': 'srk',
        'components': {'methane': 0.50, 'ethane': 0.10, 'propane': 0.10,
                       'n-butane': 0.10, 'n-pentane': 0.10, 'n-hexane': 0.10},
        'composition_basis': 'mole_fraction',
        'temperature_C': 25.0,
        'pressure_bara': 50.0,
        'total_flow': 10000,
        'flow_unit': 'kg/hr',
    },
    'process': [
        {'name': 'feed', 'type': 'stream'},
        {'name': 'separator', 'type': 'separator', 'outlet': 'liquid'},
        {'name': 'liquid valve', 'type': 'valve', 'params': {'outlet_pressure_bara': 10.0}},
    ],
}
builder2 = ProcessBuilder()
model2 = builder2.build_from_spec(spec2)
assert model2 is not None
print(f'  Units: {len(model2.list_units())}')
print('  PASSED')

print()
print('=' * 50)
print('ALL TESTS PASSED')
print('=' * 50)
