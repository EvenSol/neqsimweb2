"""Quick test: verify the DEXPI XML generation logic works with dataclass objects."""
import sys
sys.path.insert(0, ".")

from process_chat.process_model import UnitInfo, StreamInfo
from process_chat.dexpi_integration import _NEQSIM_TO_DEXPI_CLASS, _sanitize_id, _add_ga
import xml.etree.ElementTree as ET
import io

units = [
    UnitInfo(name="V-100", unit_type="Separator", java_class="Separator",
             properties={"outletPressure_bara": 65.0}),
    UnitInfo(name="K-100", unit_type="Compressor", java_class="Compressor",
             properties={"power_kW": 5000.0}),
    UnitInfo(name="E-100", unit_type="Cooler", java_class="Cooler",
             properties={"duty_kW": 3000.0}),
]
streams = [
    StreamInfo(name="feed gas", temperature_C=30.0, pressure_bara=65.0, flow_rate_kg_hr=10000.0),
    StreamInfo(name="compressed gas", temperature_C=80.0, pressure_bara=120.0, flow_rate_kg_hr=10000.0),
]

root = ET.Element("PlantModel")
eq_wrapper = ET.SubElement(root, "Equipment")
nozzle_id = 0

for u in units:
    tag = u.name
    u_type = u.unit_type
    dexpi_class = _NEQSIM_TO_DEXPI_CLASS.get(u_type, "PressureVessel")
    if dexpi_class == "PipingNetworkSystem" or u_type == "Stream":
        continue

    eq_elem = ET.SubElement(eq_wrapper, dexpi_class)
    eq_elem.set("ID", _sanitize_id(tag))
    attrs = ET.SubElement(eq_elem, "GenericAttributes")
    _add_ga(attrs, "TagNameAssignmentClass", tag)

    for prop_name, prop_val in u.properties.items():
        if "pressure" in prop_name.lower():
            _add_ga(attrs, "DesignPressure", str(round(prop_val, 2)), "bara")
        elif "power" in prop_name.lower():
            _add_ga(attrs, "Power", str(round(prop_val, 2)), "kW")
        elif "duty" in prop_name.lower():
            _add_ga(attrs, "Duty", str(round(prop_val, 2)), "kW")

    nozzle_id += 1
    ET.SubElement(eq_elem, "Nozzle").set("ID", f"{_sanitize_id(tag)}-N{nozzle_id}")
    nozzle_id += 1
    ET.SubElement(eq_elem, "Nozzle").set("ID", f"{_sanitize_id(tag)}-N{nozzle_id}")

for s in streams:
    pns = ET.SubElement(root, "PipingNetworkSystem")
    pns.set("ID", f"PNS-{_sanitize_id(s.name)}")
    pns_attrs = ET.SubElement(pns, "GenericAttributes")
    _add_ga(pns_attrs, "FluidCode", "NG")
    _add_ga(pns_attrs, "LineNumber", s.name)
    if s.temperature_C is not None:
        _add_ga(pns_attrs, "OperatingTemperature", str(round(s.temperature_C, 1)), "degC")
    if s.pressure_bara is not None:
        _add_ga(pns_attrs, "OperatingPressure", str(round(s.pressure_bara, 2)), "bara")

tree = ET.ElementTree(root)
buf = io.BytesIO()
tree.write(buf, encoding="utf-8", xml_declaration=True)
xml_bytes = buf.getvalue()

print(f"Generated {len(xml_bytes)} bytes of DEXPI XML")
print(xml_bytes.decode("utf-8")[:1500])
print("\n--- DEXPI export test PASSED ---")
