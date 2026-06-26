import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from neqsim import jneqsim
from theme import apply_theme

st.set_page_config(
    page_title="TEG Dehydration Emissions",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()

# ---------------------------------------------------------------------------
# NeqSim classes
# ---------------------------------------------------------------------------
SystemSrkCPA = jneqsim.thermo.system.SystemSrkCPAstatoil
ProcessSystem = jneqsim.process.processmodel.ProcessSystem
Stream = jneqsim.process.equipment.stream.Stream
StreamSaturatorUtil = jneqsim.process.equipment.util.StreamSaturatorUtil
Heater = jneqsim.process.equipment.heatexchanger.Heater
HeatExchanger = jneqsim.process.equipment.heatexchanger.HeatExchanger
SimpleTEGAbsorber = jneqsim.process.equipment.absorber.SimpleTEGAbsorber
WaterStripperColumn = jneqsim.process.equipment.absorber.WaterStripperColumn
DistillationColumn = jneqsim.process.equipment.distillation.DistillationColumn
Separator = jneqsim.process.equipment.separator.Separator
ThrottlingValve = jneqsim.process.equipment.valve.ThrottlingValve
Filter = jneqsim.process.equipment.filter.Filter
Pump = jneqsim.process.equipment.pump.Pump
Mixer = jneqsim.process.equipment.mixer.Mixer
Calculator = jneqsim.process.equipment.util.Calculator
Recycle = jneqsim.process.equipment.util.Recycle
WaterDewPointAnalyser = jneqsim.process.measurementdevice.WaterDewPointAnalyser

# Gas/inert feed components (water and TEG are appended automatically and kept last)
GAS_COMPONENTS = [
    'nitrogen', 'CO2', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane',
    'i-pentane', 'n-pentane', 'n-hexane', 'benzene',
]
DEFAULT_FEED = [0.245, 3.4, 85.7, 5.981, 2.743, 0.37, 0.77, 0.142, 0.166, 0.06, 0.01]

# NMVOC = non-methane volatile organic compounds (hydrocarbons incl. BTEX/benzene)
NMVOC = {'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane',
         'n-pentane', 'n-hexane', 'benzene'}
GHG_CH4 = {'methane'}


# ---------------------------------------------------------------------------
# Plant builder (adapted from the teg_dehydration_emissions notebook)
# ---------------------------------------------------------------------------
def build_teg_plant(feed_fractions, feed_flow_MSm3_day, feed_temp_C, feed_pressure_bara,
                    absorber_pressure_bara, absorber_temp_C, teg_flow_kg_hr, teg_feed_temp_C,
                    lean_teg_purity, flash_drum_pressure_bara, reboiler_temp_C,
                    stripping_gas_Sm3_hr, n_absorber_stages, stage_efficiency,
                    stripping_gas_temp_C=78.3, n_stripper_stages=2,
                    stripper_stage_efficiency=1.0, condenser_temp_C=85.0,
                    hx_rich1_UA=2224.0, hx_leanrich_UA=8316.0,
                    water_mode='saturated', water_content_ppm_mol=None,
                    saturation_temp_C=None, saturation_pressure_bara=None):
    """Build the TEG dehydration + regeneration plant with configurable inputs.

    Water content of the feed gas can be set two ways via ``water_mode``:

    * ``'saturated'`` (default) -- the dry gas is water-saturated with
      :class:`StreamSaturatorUtil`. By default it saturates at the feed
      temperature/pressure; pass ``saturation_temp_C`` / ``saturation_pressure_bara``
      to saturate at different conditions.
    * ``'specified'`` -- the feed water content is set directly from
      ``water_content_ppm_mol`` (mol-ppm) and no saturator is used.
    """
    p = ProcessSystem()

    # Component order: gas components + water + TEG (water/TEG always last two)
    n_comp = len(GAS_COMPONENTS) + 2

    feedGas = SystemSrkCPA()
    total_gas = sum(float(f) for f in feed_fractions)
    for name, frac in zip(GAS_COMPONENTS, feed_fractions):
        feedGas.addComponent(name, float(frac))
    if water_mode == 'specified':
        # mol-ppm of water relative to the total gas moles
        water_moles = (float(water_content_ppm_mol or 0.0) * 1.0e-6) * total_gas
        feedGas.addComponent('water', water_moles)
    else:
        feedGas.addComponent('water', 0.0)
    feedGas.addComponent('TEG', 0.0)
    feedGas.setMixingRule(10)
    feedGas.setMultiPhaseCheck(False)
    feedGas.init(0)

    dryFeedGas = Stream('dry feed gas', feedGas)
    dryFeedGas.setFlowRate(feed_flow_MSm3_day, 'MSm3/day')
    dryFeedGas.setTemperature(feed_temp_C, 'C')
    dryFeedGas.setPressure(feed_pressure_bara, 'bara')
    p.add(dryFeedGas)

    if water_mode == 'specified':
        # Water already present in the feed composition; skip the saturator.
        wetFeedGas = dryFeedGas
    elif saturation_temp_C is not None or saturation_pressure_bara is not None:
        # Saturate at user-specified conditions (default to feed T/P if one omitted).
        sat_T = saturation_temp_C if saturation_temp_C is not None else feed_temp_C
        sat_P = saturation_pressure_bara if saturation_pressure_bara is not None else feed_pressure_bara
        satSetter = Heater('saturation TP setter', dryFeedGas)
        satSetter.setOutTemperature(sat_T, 'C')
        satSetter.setOutPressure(sat_P, 'bara')
        p.add(satSetter)
        gasToSat = Stream('gas at saturation conditions', satSetter.getOutletStream())
        p.add(gasToSat)
        saturator = StreamSaturatorUtil('water saturator', gasToSat)
        p.add(saturator)
        wetFeedGas = Stream('water saturated feed gas', saturator.getOutletStream())
        p.add(wetFeedGas)
    else:
        saturator = StreamSaturatorUtil('water saturator', dryFeedGas)
        p.add(saturator)
        wetFeedGas = Stream('water saturated feed gas', saturator.getOutletStream())
        p.add(wetFeedGas)

    feedTPsetter = Heater('TP of gas to absorber', wetFeedGas)
    feedTPsetter.setOutPressure(absorber_pressure_bara, 'bara')
    feedTPsetter.setOutTemperature(absorber_temp_C, 'C')
    p.add(feedTPsetter)
    feedToAbsorber = Stream('feed to TEG absorber', feedTPsetter.getOutletStream())
    p.add(feedToAbsorber)

    # Lean TEG feed: water = (1 - purity), TEG = purity (mole basis)
    feedTEG = feedGas.clone()
    leanComp = [0.0] * n_comp
    leanComp[-2] = 1.0 - lean_teg_purity  # water
    leanComp[-1] = lean_teg_purity        # TEG
    feedTEG.setMolarComposition(leanComp)
    TEGFeed = Stream('TEG feed', feedTEG)
    TEGFeed.setFlowRate(teg_flow_kg_hr, 'kg/hr')
    TEGFeed.setTemperature(teg_feed_temp_C, 'C')
    TEGFeed.setPressure(absorber_pressure_bara, 'bara')
    p.add(TEGFeed)

    absorber = SimpleTEGAbsorber('TEG absorber')
    absorber.addGasInStream(feedToAbsorber)
    absorber.addSolventInStream(TEGFeed)
    absorber.setNumberOfStages(int(n_absorber_stages))
    absorber.setStageEfficiency(stage_efficiency)
    absorber.setInternalDiameter(2.240)
    p.add(absorber)

    dehydratedGas = Stream('dry gas from absorber', absorber.getGasOutStream())
    p.add(dehydratedGas)
    richTEG = Stream('rich TEG from absorber', absorber.getLiquidOutStream())
    p.add(richTEG)

    waterDewAnalyser = WaterDewPointAnalyser('water dew point analyser', dehydratedGas)
    waterDewAnalyser.setReferencePressure(feed_pressure_bara)
    p.add(waterDewAnalyser)

    flashValve = ThrottlingValve('Rich TEG HP flash valve', richTEG)
    flashValve.setOutletPressure(flash_drum_pressure_bara)
    p.add(flashValve)

    richPreheat = Heater('rich TEG preheater', flashValve.getOutletStream())
    p.add(richPreheat)

    heatEx2 = HeatExchanger('rich TEG heat exchanger 1', richPreheat.getOutletStream())
    heatEx2.setGuessOutTemperature(273.15 + 62.0)
    heatEx2.setUAvalue(hx_rich1_UA)
    p.add(heatEx2)

    flashSep = Separator('degassing separator', heatEx2.getOutStream(0))
    flashSep.setInternalDiameter(1.2)
    p.add(flashSep)
    flashGas = Stream('gas from degassing separator', flashSep.getGasOutStream())
    p.add(flashGas)
    flashLiquid = Stream('liquid from degassing separator', flashSep.getLiquidOutStream())
    p.add(flashLiquid)

    fineFilter = Filter('TEG fine filter', flashLiquid)
    fineFilter.setDeltaP(0.0, 'bara')
    p.add(fineFilter)

    heatEx = HeatExchanger('lean/rich TEG heat-exchanger', fineFilter.getOutletStream())
    heatEx.setGuessOutTemperature(273.15 + 130.0)
    heatEx.setUAvalue(hx_leanrich_UA)
    p.add(heatEx)

    flashValve2 = ThrottlingValve('Rich TEG LP flash valve', heatEx.getOutStream(0))
    flashValve2.setOutletPressure(1.2)
    p.add(flashValve2)

    stripGas = feedGas.clone()
    strippingGas = Stream('stripGas', stripGas)
    strippingGas.setFlowRate(stripping_gas_Sm3_hr, 'Sm3/hr')
    strippingGas.setTemperature(stripping_gas_temp_C, 'C')
    strippingGas.setPressure(1.2, 'bara')
    p.add(strippingGas)
    gasToReboiler = strippingGas.clone('gas to reboiler')
    p.add(gasToReboiler)

    column = DistillationColumn('TEG regeneration column', 1, True, True)
    column.setTemperatureTolerance(5.0e-2)
    column.setMassBalanceTolerance(2.0e-1)
    column.setEnthalpyBalanceTolerance(2.0e-1)
    column.addFeedStream(flashValve2.getOutletStream(), 1)
    column.getReboiler().setOutTemperature(273.15 + reboiler_temp_C)
    column.getCondenser().setOutTemperature(273.15 + condenser_temp_C)
    column.getTray(1).addStream(gasToReboiler)
    column.setTopPressure(1.2)
    column.setBottomPressure(1.2)
    column.setInternalDiameter(0.56)
    p.add(column)

    coolerRegenGas = Heater('regen gas cooler', column.getGasOutStream())
    coolerRegenGas.setOutTemperature(273.15 + 47.0)
    p.add(coolerRegenGas)

    sepRegenGas = Separator('regen gas separator', coolerRegenGas.getOutletStream())
    p.add(sepRegenGas)
    stillVent = Stream('still vent to atmosphere', sepRegenGas.getGasOutStream())
    p.add(stillVent)
    waterToTreatment = Stream('water to treatment', sepRegenGas.getLiquidOutStream())
    p.add(waterToTreatment)

    stripper = WaterStripperColumn('TEG stripper')
    stripper.addSolventInStream(column.getLiquidOutStream())
    stripper.addGasInStream(strippingGas)
    stripper.setNumberOfStages(int(n_stripper_stages))
    stripper.setStageEfficiency(stripper_stage_efficiency)
    p.add(stripper)

    recycleStripGas = Recycle('stripping gas recirc')
    recycleStripGas.addStream(stripper.getGasOutStream())
    recycleStripGas.setOutletStream(gasToReboiler)
    p.add(recycleStripGas)

    heatEx.setFeedStream(1, stripper.getLiquidOutStream())

    bufferTank = Heater('TEG buffer tank', heatEx.getOutStream(1))
    bufferTank.setOutTemperature(273.15 + 90.5)
    p.add(bufferTank)

    leanPumpLP = Pump('lean TEG LP pump', bufferTank.getOutletStream())
    leanPumpLP.setOutletPressure(3.0)
    leanPumpLP.setIsentropicEfficiency(0.75)
    p.add(leanPumpLP)

    heatEx2.setFeedStream(1, leanPumpLP.getOutletStream())

    coolerLeanTEG = Heater('lean TEG cooler', heatEx2.getOutStream(1))
    coolerLeanTEG.setOutTemperature(273.15 + teg_feed_temp_C)
    p.add(coolerLeanTEG)

    leanPumpHP = Pump('lean TEG HP pump', coolerLeanTEG.getOutletStream())
    leanPumpHP.setOutletPressure(absorber_pressure_bara)
    leanPumpHP.setIsentropicEfficiency(0.75)
    p.add(leanPumpHP)

    leanTEGtoAbs = Stream('lean TEG to absorber', leanPumpHP.getOutletStream())
    p.add(leanTEGtoAbs)

    pureTEG = feedGas.clone()
    makeupComp = [0.0] * n_comp
    makeupComp[-1] = 1.0  # TEG
    pureTEG.setMolarComposition(makeupComp)
    makeupTEG = Stream('makeup TEG', pureTEG)
    makeupTEG.setFlowRate(1e-6, 'kg/hr')
    makeupTEG.setTemperature(teg_feed_temp_C, 'C')
    makeupTEG.setPressure(absorber_pressure_bara, 'bara')
    p.add(makeupTEG)

    makeupCalc = Calculator('TEG makeup calculator')
    makeupCalc.addInputVariable(dehydratedGas)
    makeupCalc.addInputVariable(flashGas)
    makeupCalc.addInputVariable(stillVent)
    makeupCalc.addInputVariable(waterToTreatment)
    makeupCalc.setOutputVariable(makeupTEG)
    p.add(makeupCalc)

    makeupMixer = Mixer('makeup mixer')
    makeupMixer.addStream(leanTEGtoAbs)
    makeupMixer.addStream(makeupTEG)
    p.add(makeupMixer)

    recycleLeanTEG = Recycle('lean TEG recycle')
    recycleLeanTEG.addStream(makeupMixer.getOutletStream())
    recycleLeanTEG.setOutletStream(TEGFeed)
    recycleLeanTEG.setPriority(200)
    recycleLeanTEG.setDownstreamProperty('flow rate')
    p.add(recycleLeanTEG)

    richPreheat.setEnergyStream(column.getCondenser().getEnergyStream())

    streams = {
        'dehydratedGas': dehydratedGas,
        'richTEG': richTEG,
        'flashGas': flashGas,
        'stillVent': stillVent,
        'waterToTreatment': waterToTreatment,
        'leanTEGtoAbs': leanTEGtoAbs,
        'waterDewAnalyser': waterDewAnalyser,
        'column': column,
    }
    return p, streams


def run_plant(process, timeout_ms=300000):
    """Run the process on a worker thread (robust for recycle convergence)."""
    thr = process.runAsThread()
    thr.join(timeout_ms)
    return process


def comp_mass_flows_kg_hr(stream):
    fluid = stream.getFluid()
    total = stream.getFlowRate('kg/hr')
    n = fluid.getNumberOfComponents()
    zM, names = [], []
    for i in range(n):
        c = fluid.getComponent(i)
        names.append(str(c.getComponentName()))
        zM.append(c.getz() * c.getMolarMass())
    s = sum(zM)
    if s <= 0:
        return {names[i]: 0.0 for i in range(n)}
    return {names[i]: (zM[i] / s) * total for i in range(n)}


def teg_mass_fraction(stream):
    flows = comp_mass_flows_kg_hr(stream)
    tot = sum(flows.values())
    return 100.0 * flows.get('TEG', 0.0) / tot if tot > 0 else 0.0


def classify_emissions(stream):
    flows = comp_mass_flows_kg_hr(stream)
    out = {
        'NMVOC': sum(v for k, v in flows.items() if k in NMVOC),
        'methane': sum(v for k, v in flows.items() if k in GHG_CH4),
        'CO2': flows.get('CO2', 0.0),
        'nitrogen': flows.get('nitrogen', 0.0),
        'water': flows.get('water', 0.0),
        'TEG': flows.get('TEG', 0.0),
    }
    out['total'] = sum(flows.values())
    out['benzene'] = flows.get('benzene', 0.0)
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌍 TEG Dehydration — Regeneration Emission Calculator")
st.markdown("""
Model a full **triethylene-glycol (TEG) dehydration plant** (absorber → flash drum →
lean/rich exchangers → regeneration still with reboiler & stripping gas → closed TEG recycle)
with the **CPA equation of state**, and quantify the **atmospheric emissions** from the
regeneration vents — **NMVOC**, **methane (CH₄)**, **CO₂**, water and TEG loss.

Adjust any input below, then press **Run emission calculation**.
""")

with st.expander("📖 About this model — process, parameters & methodology"):
    st.markdown("""
### Process flow
1. **Absorber** — wet feed gas contacts lean TEG counter-currently in a
   `SimpleTEGAbsorber`; dry gas leaves the top, water-rich (*rich*) TEG leaves
   the bottom. Water dew point of the dry gas is measured with a
   `WaterDewPointAnalyser`.
2. **Flash drum** — rich TEG is let down to the *flash-drum pressure* and
   degassed in a separator. The released **flash gas** carries most of the
   co-absorbed hydrocarbons and is normally **recovered to fuel**.
3. **Lean/rich heat exchangers** — the cold rich TEG is pre-heated against the
   hot regenerated lean TEG to cut reboiler duty (`HeatExchanger`, set by *UA*).
4. **Regeneration still** — a `DistillationColumn` with a reboiler boils the
   water out of the TEG. The overhead **still vent** is the main **atmospheric
   emission** (water + stripped hydrocarbons + a little TEG).
5. **Stripping column** — hot lean TEG is contacted with a small dry
   stripping-gas stream (`WaterStripperColumn`) to push lean TEG purity above
   the reboiler-only limit.
6. **Closed recycle** — pumps and a make-up `Calculator` return regenerated lean
   TEG to the absorber; the `Recycle` loop is solved to convergence on a worker
   thread.

### What each control does
| Control | Effect |
|---|---|
| Feed flow / T / P | Sets the gas duty and absorber loading. |
| Absorber P / T, stages, stage efficiency | Higher P, lower T, more stages → drier gas. |
| Lean TEG circulation & feed T | More/colder lean TEG → drier gas, more reboiler duty. |
| Flash-drum pressure | **Lower → more NMVOC recovered as flash gas** instead of vented. |
| Reboiler temperature | Higher → higher lean TEG purity, more overhead vent. |
| Stripping gas rate / T / stages / efficiency | Raise lean TEG purity beyond reboiler limit. |
| HX UA values | Higher UA → more heat recovery, lower reboiler duty. |
| Condenser temperature | Reflux temperature at the still top. |

### Emission accounting
- **NMVOC** = non-methane volatile organic compounds
  (ethane … n-hexane + benzene).
- **GHG** reported separately as methane (CH₄) and CO₂.
- The **still vent** is the atmospheric release; the **flash gas** is normally
  routed to the fuel/recovery system.
- **Lean TEG purity is a calculated result** of the regeneration train and the
  recycle loop — not a design input (see *Advanced solver settings*).

### Thermodynamics
CPA EOS (`SystemSrkCPAstatoil`, mixing rule 10) — required for the strongly
non-ideal water/TEG/hydrocarbon system. Water and TEG are added internally and
kept as the last two components.
""")

st.divider()

# --- Feed gas composition ---
with st.expander("📋 Feed gas composition (molar — auto-normalized)", expanded=True):
    if 'teg_feed_df' not in st.session_state:
        st.session_state.teg_feed_df = pd.DataFrame({
            'Component': GAS_COMPONENTS,
            'MolarComposition[-]': DEFAULT_FEED,
        })
    if st.button('Reset to default composition'):
        st.session_state.teg_feed_df = pd.DataFrame({
            'Component': GAS_COMPONENTS,
            'MolarComposition[-]': DEFAULT_FEED,
        })
        st.rerun()

    feed_df = st.data_editor(
        st.session_state.teg_feed_df,
        column_config={
            'Component': st.column_config.TextColumn('Component', disabled=True),
            'MolarComposition[-]': st.column_config.NumberColumn(
                'Molar Composition [-]', min_value=0.0, max_value=100.0, format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.session_state.teg_feed_df = feed_df
    st.caption("💡 Water and TEG are added internally and kept as the last two components.")

# --- Operating parameters ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Feed gas")
    feed_flow = st.number_input("Feed flow [MSm³/day]", 0.1, 50.0, 4.65, 0.05)
    feed_temp = st.number_input("Feed temperature [°C]", -20.0, 80.0, 25.0, 0.5)
    feed_pressure = st.number_input("Feed pressure [bara]", 10.0, 200.0, 70.0, 1.0)

    water_mode_label = st.radio(
        "Feed water content",
        ["Water-saturated", "Specify water content"],
        help="Choose whether the feed gas is water-saturated (by NeqSim) or its "
             "water content is set directly.",
    )
    if water_mode_label == "Water-saturated":
        water_mode = 'saturated'
        water_content_ppm = None
        custom_sat = st.checkbox(
            "Saturate at custom T/P",
            help="By default the gas is saturated at the feed temperature and "
                 "pressure. Tick to saturate at different conditions.")
        if custom_sat:
            sat_temp = st.number_input("Saturation temperature [°C]", -20.0, 80.0,
                                       float(feed_temp), 0.5)
            sat_pressure = st.number_input("Saturation pressure [bara]", 10.0, 200.0,
                                           float(feed_pressure), 1.0)
        else:
            sat_temp = None
            sat_pressure = None
    else:
        water_mode = 'specified'
        sat_temp = None
        sat_pressure = None
        water_content_ppm = st.number_input(
            "Water content [mol-ppm]", 0.0, 5000.0, 1000.0, 10.0,
            help="Water mole-fraction of the feed gas in parts-per-million (molar). "
                 "Typical wet feed gas is several hundred to a few thousand mol-ppm.")

with col2:
    st.subheader("Absorber / TEG")
    absorber_pressure = st.number_input("Absorber pressure [bara]", 10.0, 200.0, 85.0, 1.0)
    absorber_temp = st.number_input("Gas-to-absorber temperature [°C]", 10.0, 70.0, 35.0, 0.5)
    teg_flow = st.number_input("Lean TEG circulation [kg/hr]", 500.0, 50000.0, 5500.0, 100.0)
    teg_feed_temp = st.number_input("Lean TEG feed temperature [°C]", 20.0, 80.0, 48.5, 0.5)
    n_stages = st.number_input("Absorber stages", 2, 10, 4, 1)
    stage_eff = st.slider("Stage efficiency [-]", 0.3, 1.0, 0.7, 0.05)

with col3:
    st.subheader("Regeneration")
    flash_pressure = st.number_input("Flash-drum pressure [bara]", 1.5, 15.0, 4.8, 0.1,
                                     help="Lowering this shifts NMVOC from the atmospheric "
                                          "still vent into the recovered flash gas.")
    reboiler_temp = st.number_input("Reboiler temperature [°C]", 150.0, 210.0, 197.5, 0.5)
    stripping_gas = st.number_input("Stripping-gas rate [Sm³/hr]", 0.0, 1000.0, 180.0, 10.0)

with st.expander("🌬️ Stripping column (TEG enhancement)"):
    st.caption(
        "The stripping column contacts hot lean TEG from the reboiler with a small dry "
        "stripping-gas stream to drive off the last of the water, boosting lean TEG "
        "purity above the reboiler-only limit (~98.5–99 wt%). More gas, higher "
        "temperature and more stages all increase purity — at the cost of extra gas "
        "use and NMVOC sent to the still vent."
    )
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        stripping_gas_temp = st.number_input(
            "Stripping-gas temperature [°C]", 40.0, 120.0, 78.3, 0.5,
            help="Temperature of the dry gas fed to the stripper / reboiler.")
    with sc2:
        n_stripper_stages = st.number_input("Stripper stages", 1, 6, 2, 1)
    with sc3:
        stripper_stage_eff = st.slider("Stripper stage efficiency [-]", 0.3, 1.0, 1.0, 0.05)

with st.expander("⚙️ Heat exchangers & condenser"):
    st.caption(
        "UA = overall heat-transfer coefficient × area [W/K]. Higher UA recovers more "
        "heat from the hot lean TEG into the cold rich TEG, lowering reboiler duty. "
        "The condenser (reflux) temperature sets how much water/hydrocarbon is "
        "refluxed at the top of the regeneration column."
    )
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        hx_rich1_ua = st.number_input(
            "Rich TEG HX-1 UA [W/K]", 100.0, 20000.0, 2224.0, 50.0,
            help="Pre-heat of rich TEG before the degassing separator.")
    with hc2:
        hx_leanrich_ua = st.number_input(
            "Lean/rich TEG HX UA [W/K]", 500.0, 40000.0, 8316.0, 100.0,
            help="Main lean/rich cross-exchanger feeding the regeneration column.")
    with hc3:
        condenser_temp = st.number_input(
            "Column condenser temperature [°C]", 60.0, 110.0, 85.0, 1.0)

with st.expander("Advanced solver settings"):
    st.caption(
        "The lean TEG purity is **calculated** by the regeneration train (reboiler "
        "temperature, stripping gas and flash conditions) and the closed TEG recycle "
        "loop — it is reported as a result below. The value here is only the initial "
        "composition guess that seeds the recycle; it does not fix the result, but a "
        "realistic seed helps the loop converge faster."
    )
    lean_purity = st.slider(
        "Lean TEG purity — recycle initial guess [mol TEG fraction]",
        0.90, 0.999, 0.97, 0.001,
    )

st.divider()

run_col, sweep_col = st.columns([1, 1])
with run_col:
    run_clicked = st.button("▶️ Run emission calculation", type="primary",
                            use_container_width=True)
with sweep_col:
    do_sweep = st.checkbox("Also run flash-drum pressure sensitivity sweep",
                           help="Rebuilds and solves the plant at several flash-drum "
                                "pressures (2–9 bara). Slower.")


def _fractions():
    return [float(v) for v in st.session_state.teg_feed_df['MolarComposition[-]'].tolist()]


def _build_kwargs(flash_p):
    return dict(
        feed_fractions=_fractions(),
        feed_flow_MSm3_day=feed_flow,
        feed_temp_C=feed_temp,
        feed_pressure_bara=feed_pressure,
        absorber_pressure_bara=absorber_pressure,
        absorber_temp_C=absorber_temp,
        teg_flow_kg_hr=teg_flow,
        teg_feed_temp_C=teg_feed_temp,
        lean_teg_purity=lean_purity,
        flash_drum_pressure_bara=flash_p,
        reboiler_temp_C=reboiler_temp,
        stripping_gas_Sm3_hr=stripping_gas,
        n_absorber_stages=n_stages,
        stage_efficiency=stage_eff,
        stripping_gas_temp_C=stripping_gas_temp,
        n_stripper_stages=n_stripper_stages,
        stripper_stage_efficiency=stripper_stage_eff,
        condenser_temp_C=condenser_temp,
        hx_rich1_UA=hx_rich1_ua,
        hx_leanrich_UA=hx_leanrich_ua,
        water_mode=water_mode,
        water_content_ppm_mol=water_content_ppm,
        saturation_temp_C=sat_temp,
        saturation_pressure_bara=sat_pressure,
    )


if run_clicked:
    if sum(_fractions()) <= 0:
        st.error("Feed composition sums to zero — enter at least one component.")
        st.stop()
    try:
        with st.spinner("Building and solving the TEG plant (recycle convergence)…"):
            process, S = build_teg_plant(**_build_kwargs(flash_pressure))
            run_plant(process)

            water_dew_C = float(S['waterDewAnalyser'].getMeasuredValue('C'))
            lean_teg_wt = teg_mass_fraction(S['leanTEGtoAbs'])
            dry_gas_flow = float(S['dehydratedGas'].getFlowRate('MSm3/day'))
            still = classify_emissions(S['stillVent'])
            flash = classify_emissions(S['flashGas'])
    except Exception as e:
        st.error(f"Calculation failed: {e}")
        st.stop()

    st.success("Calculation complete.")

    # KPIs
    st.subheader("Dehydration performance")
    k1, k2, k3 = st.columns(3)
    k1.metric("Dry-gas water dew point", f"{water_dew_C:.2f} °C",
              help=f"At {feed_pressure:.0f} bara reference")
    k2.metric("Lean TEG purity", f"{lean_teg_wt:.2f} wt%")
    k3.metric("Dry gas flow", f"{dry_gas_flow:.3f} MSm³/day")

    # Emission KPIs (atmospheric still vent)
    st.subheader("Atmospheric emissions — still vent")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("NMVOC", f"{still['NMVOC']:.3f} kg/hr",
              help=f"{still['NMVOC']*24/1000:.3f} t/day")
    e2.metric("Methane (CH₄)", f"{still['methane']:.3f} kg/hr")
    e3.metric("CO₂", f"{still['CO2']:.3f} kg/hr")
    e4.metric("Benzene (BTEX)", f"{still['benzene']:.4f} kg/hr")

    # Emission table
    st.subheader("Vent emission breakdown [kg/hr]")
    groups = ['total', 'NMVOC', 'methane', 'CO2', 'water', 'TEG', 'benzene']
    table = pd.DataFrame({
        'Group': groups,
        'Still vent (atmospheric)': [still[g] for g in groups],
        'Flash gas (to fuel)': [flash[g] for g in groups],
    })
    table['Still vent (t/day)'] = table['Still vent (atmospheric)'] * 24 / 1000.0
    st.dataframe(
        table.style.format({
            'Still vent (atmospheric)': '{:.4f}',
            'Flash gas (to fuel)': '{:.4f}',
            'Still vent (t/day)': '{:.4f}',
        }),
        hide_index=True, use_container_width=True,
    )

    # Bar chart
    plot_groups = ['NMVOC', 'methane', 'CO2', 'water', 'TEG']
    fig = go.Figure()
    fig.add_bar(name='Still vent (atmospheric)', x=plot_groups,
                y=[still[g] for g in plot_groups], marker_color='#c0392b')
    fig.add_bar(name='Flash gas (to fuel)', x=plot_groups,
                y=[flash[g] for g in plot_groups], marker_color='#2980b9')
    fig.update_layout(
        barmode='group', yaxis_title='Mass flow [kg/hr]',
        title=f'TEG regeneration vent emissions (flash drum {flash_pressure:.1f} bara)',
        legend=dict(orientation='h', y=1.12), height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Downloadable results
    st.download_button(
        "⬇️ Download emission table (CSV)",
        table.to_csv(index=False).encode('utf-8'),
        file_name="teg_emissions.csv", mime="text/csv",
    )

    # Optional sensitivity sweep
    if do_sweep:
        st.divider()
        st.subheader("Flash-drum pressure sensitivity")
        sweep_pressures = [2.0, 3.0, 4.8, 7.0, 9.0]
        still_nmvoc, flash_nmvoc = [], []
        prog = st.progress(0.0, text="Running sweep…")
        try:
            for i, pf in enumerate(sweep_pressures):
                proc, Ss = build_teg_plant(**_build_kwargs(pf))
                run_plant(proc)
                st_em = classify_emissions(Ss['stillVent'])
                fl_em = classify_emissions(Ss['flashGas'])
                still_nmvoc.append(st_em['NMVOC'])
                flash_nmvoc.append(fl_em['NMVOC'])
                prog.progress((i + 1) / len(sweep_pressures),
                              text=f"Solved flash drum {pf:.1f} bara")
            prog.empty()

            sfig = go.Figure()
            sfig.add_scatter(x=sweep_pressures, y=still_nmvoc, mode='lines+markers',
                             name='Still-vent NMVOC (atmospheric)',
                             line=dict(color='#c0392b'))
            sfig.add_scatter(x=sweep_pressures, y=flash_nmvoc, mode='lines+markers',
                             name='Flash-gas NMVOC (recovered to fuel)',
                             line=dict(color='#2980b9', dash='dash'))
            sfig.update_layout(
                xaxis_title='Flash-drum pressure [bara]',
                yaxis_title='NMVOC mass flow [kg/hr]',
                title='Effect of flash-drum pressure on TEG regeneration emissions',
                legend=dict(orientation='h', y=1.12), height=450,
            )
            st.plotly_chart(sfig, use_container_width=True)

            hi = still_nmvoc[sweep_pressures.index(9.0)]
            lo = still_nmvoc[sweep_pressures.index(2.0)]
            if hi > 0:
                st.info(f"Lowering the flash drum from 9.0 → 2.0 bara reduces atmospheric "
                        f"still-vent NMVOC from {hi:.3f} to {lo:.3f} kg/hr "
                        f"(**{100.0*(hi-lo)/hi:.1f}% reduction**).")
        except Exception as e:
            prog.empty()
            st.warning(f"Sensitivity sweep failed: {e}")
else:
    st.info("Set the inputs above and press **Run emission calculation**.")

st.divider()
st.caption("Model: CPA EOS (SystemSrkCPAstatoil, mixing rule 10). NMVOC = non-methane "
           "volatile organic compounds. The still vent is the atmospheric emission; "
           "flash gas is normally recovered to fuel.")
