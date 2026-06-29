"""Test whether stream flow scaling persists through ProcessSystem.run()"""
import pathlib
from process_chat.process_model import NeqSimProcessModel

p = pathlib.Path("test_process.neqsim")

# Test 1: Run base case
m1 = NeqSimProcessModel.from_file(str(p))
r1 = m1.run()
base_power = r1.kpis.get("1st stage compressor.power_kW")
feed_flow = r1.kpis.get("feed gas.flow_kg_hr")
print("=== BASE CASE ===")
if feed_flow:
    print(f"Feed flow: {feed_flow.value:.1f} kg/hr")
if base_power:
    print(f"Compressor power: {base_power.value:.2f} kW")

# Test 2: Scale flow 2x, then run
m2 = NeqSimProcessModel.from_file(str(p))
s = m2.get_stream("feed gas")
orig_flow = float(s.getFlowRate("kg/hr"))
print(f"\n=== SCALING to 2x ===")
print(f"Original flow: {orig_flow:.1f} kg/hr")
s.setFlowRate(orig_flow * 2.0, "kg/hr")
after_set = float(s.getFlowRate("kg/hr"))
print(f"After setFlowRate: {after_set:.1f} kg/hr")

# Also check the underlying fluid
fluid = s.getThermoSystem()
if fluid:
    print(f"Fluid total moles: {float(fluid.getTotalNumberOfMoles()):.4f}")
    print(f"Fluid total flow kg/hr: {float(fluid.getFlowRate('kg/hr')):.1f}")

r2 = m2.run()
scaled_power = r2.kpis.get("1st stage compressor.power_kW")
scaled_feed = r2.kpis.get("feed gas.flow_kg_hr")
if scaled_feed:
    print(f"Feed flow after run: {scaled_feed.value:.1f} kg/hr")
if scaled_power:
    print(f"Compressor power after run: {scaled_power.value:.2f} kW")
if scaled_power and base_power:
    delta = scaled_power.value - base_power.value
    print(f"Power delta: {delta:+.2f} kW ({delta/base_power.value*100:+.1f}%)")
    print(f"Power changed? {abs(delta) > 1.0}")

# Test 3: Also try setting via thermoSystem directly
print("\n=== TEST 3: Set via thermoSystem ===")
m3 = NeqSimProcessModel.from_file(str(p))
s3 = m3.get_stream("feed gas")
fluid3 = s3.getThermoSystem()
orig3 = float(s3.getFlowRate("kg/hr"))
print(f"Original flow: {orig3:.1f}")
fluid3.setTotalFlowRate(orig3 * 2.0, "kg/hr")
print(f"After setTotalFlowRate on fluid: {float(s3.getFlowRate('kg/hr')):.1f}")

r3 = m3.run()
p3 = r3.kpis.get("1st stage compressor.power_kW")
f3 = r3.kpis.get("feed gas.flow_kg_hr")
if f3:
    print(f"Feed flow after run: {f3.value:.1f} kg/hr")
if p3:
    print(f"Compressor power: {p3.value:.2f} kW")
    if base_power:
        print(f"Power changed? {abs(p3.value - base_power.value) > 1.0}")
