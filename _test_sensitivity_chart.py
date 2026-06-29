"""Test: sensitivity chart filtering and deduplication logic."""
from neqsim import jneqsim
from process_chat.process_model import NeqSimProcessModel
from process_chat.sensitivity import _run_single_sweep

# Build process: Stream -> Cooler -> Compressor -> Aftercooler
f = jneqsim.thermo.system.SystemPrEos(273.15 + 40, 30.0)
f.addComponent('methane', 0.85)
f.addComponent('ethane', 0.06)
f.addComponent('propane', 0.04)
f.addComponent('CO2', 0.02)
f.addComponent('nitrogen', 0.03)
f.setMixingRule(2)

proc = jneqsim.process.processmodel.ProcessSystem()
feed = jneqsim.process.equipment.stream.Stream('Feed Gas', f)
feed.setFlowRate(50000, 'kg/hr')
feed.setTemperature(40, 'C')
feed.setPressure(30.0, 'bara')
proc.add(feed)

cooler = jneqsim.process.equipment.heatexchanger.Cooler('24-HA-01', feed)
cooler.setOutTemperature(25.0, 'C')
proc.add(cooler)

comp = jneqsim.process.equipment.compressor.Compressor('27-KA-01', cooler.getOutletStream())
comp.setOutletPressure(80.0)
comp.setIsentropicEfficiency(0.75)
proc.add(comp)

aftercooler = jneqsim.process.equipment.heatexchanger.Cooler('27-HA-01', comp.getOutletStream())
aftercooler.setOutTemperature(40.0, 'C')
proc.add(aftercooler)

proc.run()
model = NeqSimProcessModel.from_process_system(proc)

# Run sweep
result = _run_single_sweep(
    model,
    variable='units.24-HA-01.outtemperature_c',
    min_value=20.0,
    max_value=30.0,
    n_points=6,
    response_kpis=['power', 'duty', 'temperature'],
)

# Simulate the chart logic
pts = [p for p in result.sweep_points if p.feasible and p.output_values]
print(f'Total points: {len(result.sweep_points)}, feasible with output: {len(pts)}')

var_name = result.sweep_variable or list(pts[0].input_values.keys())[0]
x_data = [p.input_values.get(var_name, 0) for p in pts]

_SKIP = {'mechdesign', 'sizing', 'composition', 'weight fraction',
         'mole fraction', 'maxdesign', 'maxoperating', 'json.',
         'molar_volume', 'molar_mass_kg', 'jointefficiency',
         'tensilestrength', 'maxallowablestress', 'mindesign',
         'report.'}

all_kpi_keys = sorted({k for p in pts for k in p.output_values})
print(f'Total KPI keys across all points: {len(all_kpi_keys)}')

filtered = {}
for k in all_kpi_keys:
    kl = k.lower()
    if any(s in kl for s in _SKIP):
        continue
    ys = [p.output_values.get(k, 0) for p in pts]
    rng = max(ys) - min(ys)
    if rng < 1e-9:
        continue
    if all(abs(v) < 1e-12 for v in ys):
        continue
    filtered[k] = ys

print(f'After filtering noise/constant/zero/report: {len(filtered)}')
for k in sorted(filtered):
    ys = filtered[k]
    print(f'  {k}: [{ys[0]:.2f} ... {ys[-1]:.2f}]')

# Group by response_kpi
response_kpis = result.response_kpis
used_keys = set()
print()
for resp_kpi in response_kpis:
    rkl = resp_kpi.lower()
    matching = {k: ys for k, ys in filtered.items()
                if rkl in k.lower() and k not in used_keys}
    if matching:
        sorted_keys = sorted(matching, key=len)[:5]
        used_keys.update(sorted_keys)
        print(f'Chart "{resp_kpi}": {sorted_keys}')

remaining = {k for k in filtered if k not in used_keys}
if remaining:
    print(f'Remaining chart: {sorted(remaining, key=len)[:8]}')
else:
    print('No remaining ungrouped KPIs')
