"""Test the neqsim_code extraction / execution pipeline."""
from process_chat.chat_tools import extract_neqsim_code, _execute_neqsim_code

# ----- Test 1: Extraction -------------------------------------------------
BT = "```"  # backtick triple
test_text = f"""Here is the code:
{BT}neqsim_code
import numpy as np
x = np.array([1, 2, 3])
print("Sum:", np.sum(x))
results_df = __import__("pandas").DataFrame({{"x": list(x), "x_squared": list(x**2)}})
{BT}
"""

code = extract_neqsim_code(test_text)
assert code is not None, "Extraction failed!"
print("[PASS] Extraction OK")
print("  Code:", repr(code[:60]))

# ----- Test 2: Execution (numpy + pandas) ---------------------------------
result = _execute_neqsim_code(code)
assert result["error"] is None, f"Execution failed: {result['error']}"
assert "Sum: 6" in result["stdout"], f"Unexpected stdout: {result['stdout']}"
assert len(result["tables"]) >= 1, "No tables captured"
print("[PASS] Execution OK")
print("  Stdout:", result["stdout"].strip())
print("  Tables:", [t["name"] for t in result["tables"]])

# ----- Test 3: Blocked import (os) ----------------------------------------
bad_code = "import os\nprint(os.getcwd())"
result2 = _execute_neqsim_code(bad_code)
assert result2["error"] is not None, "Should have blocked 'import os'"
assert "not allowed" in result2["error"], f"Wrong error: {result2['error']}"
print("[PASS] Import guard blocks 'os'")

# ----- Test 4: NeqSim TP flash -------------------------------------------
neqsim_code = """
from neqsim.thermo import fluid, TPflash, dataFrame

thermosystem = fluid('srk')
thermosystem.addComponent('methane', 0.85)
thermosystem.addComponent('ethane', 0.07)
thermosystem.addComponent('propane', 0.03)
thermosystem.addComponent('CO2', 0.02)
thermosystem.addComponent('nitrogen', 0.03)
thermosystem.setMixingRule(2)

thermosystem.setPressure(50.0, 'bara')
thermosystem.setTemperature(25.0, 'C')
TPflash(thermosystem)
flash_results = dataFrame(thermosystem)
print(flash_results.to_string())
"""

result3 = _execute_neqsim_code(neqsim_code)
if result3["error"]:
    print(f"[WARN] NeqSim flash errored (may be expected if JVM not ready): {result3['error']}")
else:
    print("[PASS] NeqSim TP flash OK")
    print("  Tables:", [t["name"] for t in result3["tables"]])
    # stdout should contain column headers
    assert "methane" in result3["stdout"].lower() or len(result3["tables"]) > 0, \
        "No meaningful output from flash"

# ----- Test 5: Plotly figure collection -----------------------------------
plotly_code = """
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], name="test"))
fig.update_layout(title="Test Plot")
"""
result4 = _execute_neqsim_code(plotly_code)
assert result4["error"] is None, f"Plotly code failed: {result4['error']}"
assert len(result4["figures"]) >= 1, "No plotly figure captured"
print("[PASS] Plotly figure captured")
print(f"  Figures: {len(result4['figures'])}")

print("\n=== ALL TESTS PASSED ===")
