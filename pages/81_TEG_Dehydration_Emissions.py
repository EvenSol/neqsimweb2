import streamlit as st
import pandas as pd
import numpy as np
import json
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
Splitter = jneqsim.process.equipment.splitter.Splitter
Compressor = jneqsim.process.equipment.compressor.Compressor
ThrottlingValve = jneqsim.process.equipment.valve.ThrottlingValve
Filter = jneqsim.process.equipment.filter.Filter
Pump = jneqsim.process.equipment.pump.Pump
Mixer = jneqsim.process.equipment.mixer.Mixer
Calculator = jneqsim.process.equipment.util.Calculator
Recycle = jneqsim.process.equipment.util.Recycle
WaterDewPointAnalyser = jneqsim.process.measurementdevice.WaterDewPointAnalyser
GraphvizExporter = jneqsim.process.processmodel.ProcessSystemGraphvizExporter

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
                    saturation_temp_C=None, saturation_pressure_bara=None,
                    recirculate_stripping_gas=True,
                    recycle_blower_discharge_bara=1.4):
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
    waterToTreatment = Stream('water/HC to process or flare drum',
                             sepRegenGas.getLiquidOutStream())
    p.add(waterToTreatment)

    if recirculate_stripping_gas:
        # Closed-loop stripping gas: take it from the dried regenerator overhead
        # (the gas leaving the condenser knock-out drum), recondition it and feed
        # it back to the stripper. A fixed slice equal to the stripping-gas rate
        # is recirculated; only the remainder leaves as the atmospheric vent.
        recircSplit = Splitter('stripping gas recirc split',
                               sepRegenGas.getGasOutStream())
        recircSplit.setFlowRates([float(stripping_gas_Sm3_hr), -1.0], 'Sm3/hr')
        p.add(recircSplit)
        stillVent = Stream('still vent (flare/vent/recompression)',
                           recircSplit.getSplitStream(1))
        p.add(stillVent)
        # Recycle blower: the regenerator overhead is at low pressure (~1.2 bara),
        # so a blower boosts the recirculated slice up to the pressure the stripper
        # needs (regeneration pressure + line/equipment losses) before it is
        # reconditioned and looped back as stripping gas.
        recircBlower = Compressor('stripping gas recycle blower',
                                  recircSplit.getSplitStream(0))
        recircBlower.setOutletPressure(recycle_blower_discharge_bara)
        recircBlower.setIsentropicEfficiency(0.75)
        p.add(recircBlower)
        recircHeater = Heater('stripping gas recirc heater',
                              recircBlower.getOutletStream())
        recircHeater.setOutTemperature(273.15 + stripping_gas_temp_C)
        p.add(recircHeater)
    else:
        recircBlower = None
        recircHeater = None
        stillVent = Stream('still vent (flare/vent/recompression)',
                           sepRegenGas.getGasOutStream())
        p.add(stillVent)

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

    if recirculate_stripping_gas:
        # Close the stripping-gas loop: the reconditioned recirculated slice from
        # the dried overhead becomes the stripper gas feed (tear stream).
        recycleStrippingMakeup = Recycle('stripping gas makeup recycle')
        recycleStrippingMakeup.addStream(recircHeater.getOutletStream())
        recycleStrippingMakeup.setOutletStream(strippingGas)
        recycleStrippingMakeup.setPriority(150)
        p.add(recycleStrippingMakeup)

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
        'strippingGas': strippingGas,
        'column': column,
        'recircBlower': recircBlower,
    }
    return p, streams


def run_plant(process, timeout_ms=300000):
    """Run the process on a worker thread (robust for recycle convergence)."""
    thr = process.runAsThread()
    thr.join(timeout_ms)
    return process


def export_process_dot(process, temp_unit='C', pressure_unit='bara', flow_unit='kg/hr'):
    """Export the solved process topology to a Graphviz DOT string.

    Each stream edge is annotated with its temperature, pressure and mass flow
    so the connectivity (including the recycle loops) is visible. Rendered
    client-side by ``st.graphviz_chart`` — no system Graphviz binary needed.
    """
    import os
    import tempfile
    Opts = GraphvizExporter.GraphvizExportOptions
    options = (Opts.builder()
               .includeStreamTemperatures(True)
               .includeStreamPressures(True)
               .includeStreamFlowRates(True)
               .includeStreamPropertyTable(False)
               .temperatureUnit(temp_unit)
               .pressureUnit(pressure_unit)
               .flowRateUnit(flow_unit)
               .tablePlacement(Opts.TablePlacement.BELOW)
               .build())
    fd, path = tempfile.mkstemp(suffix='.dot')
    os.close(fd)
    try:
        process.exportToGraphviz(path, options)
        with open(path, 'r', encoding='utf-8') as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_process_report(process):
    """Build a JSON-serialisable report of every unit operation and its results.

    For each unit we capture its name, class, any NeqSim ``getResultTable()``
    rows, and (for material streams) the temperature, pressure, flow and overall
    molar composition. Every getter is wrapped in try/except so one unsupported
    method never breaks the whole report.
    """
    report = {'units': []}
    try:
        units = list(process.getUnitOperations())
    except Exception:
        units = []
    for u in units:
        entry = {}
        try:
            entry['name'] = str(u.getName())
        except Exception:
            entry['name'] = '(unnamed)'
        try:
            entry['type'] = str(u.getClass().getSimpleName())
        except Exception:
            entry['type'] = '?'

        # Generic NeqSim result table (name / value / unit rows) where available
        try:
            tbl = u.getResultTable()
            if tbl is not None:
                rows = []
                for row in tbl:
                    if row is None:
                        continue
                    rows.append([None if c is None else str(c) for c in row])
                if rows:
                    entry['resultTable'] = rows
        except Exception:
            pass

        # Material-stream conditions + overall molar composition
        if entry.get('type') == 'Stream':
            cond = {}
            for key, getter, args in (
                ('temperature_C', 'getTemperature', ('C',)),
                ('pressure_bara', 'getPressure', ('bara',)),
                ('flow_kg_hr', 'getFlowRate', ('kg/hr',)),
                ('flow_MSm3_day', 'getFlowRate', ('MSm3/day',)),
            ):
                try:
                    v = float(getattr(u, getter)(*args))
                    if v == v:  # not NaN
                        cond[key] = round(v, 6)
                except Exception:
                    pass
            try:
                fluid = u.getFluid()
                comp = {}
                for i in range(int(fluid.getNumberOfComponents())):
                    c = fluid.getComponent(i)
                    comp[str(c.getComponentName())] = round(float(c.getz()), 8)
                if comp:
                    cond['molar_composition'] = comp
                cond['number_of_phases'] = int(fluid.getNumberOfPhases())
            except Exception:
                pass
            if cond:
                entry['stream'] = cond

        # Common scalar results (power for rotating equipment, duty for HX/heaters)
        scal = {}
        try:
            p = float(u.getPower('kW'))
            if p == p:
                scal['power_kW'] = round(p, 4)
        except Exception:
            pass
        try:
            d = float(u.getDuty())  # W
            if d == d:
                scal['duty_kW'] = round(d / 1000.0, 4)
        except Exception:
            pass
        if scal:
            entry['results'] = scal

        report['units'].append(entry)
    return report


def _enlarge_dot_fonts(dot, graph_fs=18, node_fs=15, edge_fs=13):
    """Inject larger default fonts + spacing so the flowsheet text is readable.

    Adds global ``graph``/``node``/``edge`` font defaults right after the
    ``digraph`` opening brace. Per-element attributes set by the exporter still
    win, but most labels inherit these larger defaults.
    """
    idx = dot.find('{')
    if idx == -1:
        return dot
    inject = (
        '\n  graph [fontname="Helvetica", fontsize=%d, nodesep=0.4, ranksep=0.7];'
        '\n  node [fontname="Helvetica", fontsize=%d];'
        '\n  edge [fontname="Helvetica", fontsize=%d];' % (graph_fs, node_fs, edge_fs)
    )
    return dot[:idx + 1] + inject + dot[idx + 1:]


def render_graphviz_interactive(dot, height=720):
    """Render a DOT string with pan + mouse-wheel zoom.

    Uses **viz.js** (Graphviz compiled to a self-contained asm.js bundle — no
    separate ``.wasm`` file to fetch) to turn the DOT into an SVG, then
    **svg-pan-zoom** for smooth scroll-zoom, drag-pan and a reset/fit button.

    This is deliberately *not* d3-graphviz: d3-graphviz loads the Graphviz
    engine from a separate ``@hpcc-js/wasm`` ``.wasm`` file, and that fetch is
    frequently blocked inside the sandboxed Streamlit Cloud component iframe —
    which renders the toolbar but leaves the diagram **blank**. viz.js bundles
    the engine in the JS itself, so it works in the sandboxed iframe.
    """
    import json as _json
    dot_js = _json.dumps(dot)
    html = """
<div style="border:1px solid #ddd;border-radius:6px;overflow:hidden;background:#fff;">
  <div style="padding:6px 10px;background:#f5f5f5;border-bottom:1px solid #ddd;
              font:13px Helvetica,Arial,sans-serif;color:#333;">
    🖱️ Scroll to zoom &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp;
    <button id="resetBtn" style="font:12px Helvetica;cursor:pointer;
            padding:2px 8px;border:1px solid #bbb;border-radius:4px;background:#fff;">
      Reset / fit
    </button>
    <span id="status" style="margin-left:10px;color:#999;"></span>
  </div>
  <div id="graph" style="width:100%;height:__HEIGHT__px;overflow:hidden;"></div>
</div>
<script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
<script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
<script src="https://unpkg.com/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script>
  var dot = __DOT__;
  var statusEl = document.getElementById("status");
  function render() {
    if (typeof Viz === "undefined" || typeof svgPanZoom === "undefined") {
      setTimeout(render, 150); return;
    }
    var viz = new Viz();
    viz.renderSVGElement(dot).then(function (svg) {
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", "100%");
      svg.style.width = "100%";
      svg.style.height = "100%";
      var container = document.getElementById("graph");
      container.innerHTML = "";
      container.appendChild(svg);
      var pz = svgPanZoom(svg, {
        zoomEnabled: true, controlIconsEnabled: false, dblClickZoomEnabled: false,
        fit: true, center: true, minZoom: 0.1, maxZoom: 20,
        zoomScaleSensitivity: 0.3
      });
      document.getElementById("resetBtn").onclick = function () {
        pz.resize(); pz.fit(); pz.center();
      };
      window.addEventListener("resize", function () {
        pz.resize(); pz.fit(); pz.center();
      });
    }).catch(function (err) {
      statusEl.textContent = "diagram render error: " + err;
    });
  }
  render();
</script>
"""
    html = (html.replace('__DOT__', dot_js)
                .replace('__HEIGHT__', str(int(height))))
    import streamlit.components.v1 as components
    components.html(html, height=height + 60, scrolling=False)


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


def _classify_flows(flows):
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


def classify_emissions(stream):
    return _classify_flows(comp_mass_flows_kg_hr(stream))


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

### Stream connectivity (how the units are wired)
The plant is a single `ProcessSystem` solved with two (or three) tear/recycle
loops. Following the streams from the feed:

- **Gas train:** `dry feed gas` → *water saturator* (or T/P setter + saturator,
  or direct in *specify-water* mode) → `feed to TEG absorber` → **TEG absorber**
  → `dry gas from absorber` (→ `WaterDewPointAnalyser`).
- **Rich-TEG let-down:** absorber bottoms `rich TEG` → **HP flash valve**
  (→ flash-drum pressure) → *rich-TEG pre-heater* → **rich-TEG/lean-TEG HX-1**
  → **degassing separator** → `flash gas` (to fuel) + liquid.
- **Regeneration:** flash liquid → *TEG fine filter* → **lean/rich HX** →
  **LP flash valve** (→ 1.2 bara) → **regeneration column** feed (tray 1).
  The column reboiler boils water off; the overhead goes to the *regen-gas
  cooler* (47 °C) → **regen-gas knock-out separator** → `still vent` (gas) +
  `water/HC to process or flare drum` (liquid).
- **Stripping:** column bottoms (hot lean TEG) → **TEG stripper**, contacted
  with `stripGas`. The stripper overhead is returned to the column reboiler via
  the **`stripping gas recirc` recycle** (tear → `gas to reboiler`).
- **Lean-TEG return loop:** stripper bottoms → **lean/rich HX (hot side)** →
  *TEG buffer tank* → *LP pump* → **HX-1 (hot side)** → *lean-TEG cooler* →
  *HP pump* → `lean TEG to absorber` → make-up `Mixer` (+ make-up TEG) →
  **`lean TEG recycle`** (priority 200, tear → absorber `TEG feed`).
- **Stripping-gas recirculation (optional, closed loop):** when enabled, a
  `Splitter` on the **knock-out drum overhead** (`still vent` gas) draws a slice
  equal to the stripping-gas rate, reheats it (*stripping gas recirc heater*) and
  returns it through the **`stripping gas makeup recycle`** (priority 150, tear →
  `stripGas`). The split remainder becomes the **net** still vent.

After a run, open **🔀 Process flow diagram** to see the auto-generated
Graphviz flowsheet with every stream's temperature, pressure and mass flow.

### What each control does
| Control | Effect |
|---|---|
| Feed flow / T / P | Sets the gas duty and absorber loading. |
| Feed water content | **Water-saturated** (default) saturates the gas with water at feed conditions (or at a custom T/P), or **Specify water content** sets the inlet water directly in mol-ppm. |
| Absorber P / T, stages, stage efficiency | Higher P, lower T, more stages → drier gas. |
| Lean TEG circulation & feed T | More/colder lean TEG → drier gas, more reboiler duty. |
| Flash-drum pressure | **Lower → more NMVOC recovered as flash gas** instead of vented. |
| Reboiler temperature | Higher → higher lean TEG purity, more overhead vent. |
| Stripping gas rate / T / stages / efficiency | Raise lean TEG purity beyond reboiler limit. |
| Recirculate stripping gas | Closed-loop recycle: the stripping gas is taken from the dried regenerator overhead (after the condenser knock-out drum) and looped back to the stripper. Only the remaining liberated water + hydrocarbons leaves as the net vent. |
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
    recirculate_stripping_gas = st.checkbox(
        "♻️ Recirculate stripping gas (closed loop, recover instead of vent)",
        value=True,
        help="For plants that recirculate the stripping gas: the gas is taken from "
             "the dried regenerator overhead (after the condenser knock-out drum), "
             "reconditioned and looped back to the stripper as a true closed loop. "
             "A fixed slice equal to the stripping-gas rate is recirculated; only the "
             "remainder (water + hydrocarbons liberated from the TEG) leaves as the "
             "atmospheric vent. Adds a recycle loop, so convergence is a little slower.")
    if recirculate_stripping_gas:
        recycle_blower_discharge = st.number_input(
            "Recycle blower discharge pressure [bara]", 1.2, 5.0, 1.4, 0.05,
            help="The recycle blower boosts the low-pressure regenerator-overhead gas "
                 "(suction ≈1.2 bara) back up to the pressure the stripper needs "
                 "(regeneration pressure plus line/equipment losses) before it is "
                 "reconditioned and looped back to the stripper.")
    else:
        recycle_blower_discharge = 1.4

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
        recirculate_stripping_gas=recirculate_stripping_gas,
        recycle_blower_discharge_bara=recycle_blower_discharge,
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
            try:
                dot_graph = export_process_dot(process)
            except Exception:
                dot_graph = None

            try:
                process_report = build_process_report(process)
            except Exception:
                process_report = None

            # Builder/input JSON that can recreate the model via
            # ProcessSystem.fromJsonAndRun(...). Serialize now while the live
            # Java `process` object is in scope; persist as a plain string.
            try:
                _exporter = jneqsim.process.processmodel.JsonProcessExporter()
                model_input_json = str(_exporter.toJson(process, True))
            except Exception:
                model_input_json = None

            blower_kw = None
            if recirculate_stripping_gas:
                try:
                    blower_kw = float(S['recircBlower'].getPower('kW'))
                except Exception:
                    blower_kw = None

            sweep_data = None
            if do_sweep:
                sweep_pressures = [2.0, 3.0, 4.8, 7.0, 9.0]
                s_nmvoc, f_nmvoc = [], []
                prog = st.progress(0.0, text="Running sweep…")
                try:
                    for i, pf in enumerate(sweep_pressures):
                        proc, Ss = build_teg_plant(**_build_kwargs(pf))
                        run_plant(proc)
                        st_em = classify_emissions(Ss['stillVent'])
                        fl_em = classify_emissions(Ss['flashGas'])
                        s_nmvoc.append(st_em['NMVOC'])
                        f_nmvoc.append(fl_em['NMVOC'])
                        prog.progress((i + 1) / len(sweep_pressures),
                                      text=f"Solved flash drum {pf:.1f} bara")
                    prog.empty()
                    sweep_data = {'pressures': sweep_pressures,
                                  'still_nmvoc': s_nmvoc, 'flash_nmvoc': f_nmvoc}
                except Exception as e:
                    prog.empty()
                    st.warning(f"Sensitivity sweep failed: {e}")
    except Exception as e:
        st.error(f"Calculation failed: {e}")
        st.stop()

    # Persist results so that interacting with the diagram height / text-size
    # widgets (which trigger a Streamlit rerun with run_clicked=False) does NOT
    # wipe the page or blank the diagram.
    st.session_state['teg_results'] = {
        'water_dew_C': water_dew_C,
        'lean_teg_wt': lean_teg_wt,
        'dry_gas_flow': dry_gas_flow,
        'still': still,
        'flash': flash,
        'dot_graph': dot_graph,
        'process_report': process_report,
        'model_input_json': model_input_json,
        'recirculate': bool(recirculate_stripping_gas),
        'recycle_blower_discharge': float(recycle_blower_discharge),
        'blower_kw': blower_kw,
        'flash_pressure': float(flash_pressure),
        'feed_pressure': float(feed_pressure),
        'sweep': sweep_data,
    }

res = st.session_state.get('teg_results')
if res:
    still = res['still']
    flash = res['flash']
    flash_pressure_r = res['flash_pressure']

    st.success("Calculation complete.")
    if res['recirculate']:
        st.info("♻️ **Stripping gas recirculated** — the stripping gas is taken from "
                "the dried regenerator overhead (after the condenser knock-out drum), "
                "reconditioned and looped back to the stripper. The still vent below is "
                "the **net** atmospheric emission — the remainder after the recirculated "
                "slice is removed.")
        if res['blower_kw'] is not None:
            st.caption(f"Recycle blower duty ≈ {res['blower_kw']:.2f} kW "
                       f"(suction ≈1.2 bara → "
                       f"{res['recycle_blower_discharge']:.2f} bara discharge).")

    # Interactive process flow diagram (streams annotated with T, P, flow)
    dot_graph = res['dot_graph']
    if dot_graph:
        with st.expander("🔀 Process flow diagram (streams with T, P, flow)", expanded=False):
            st.caption("Auto-generated from the solved NeqSim `ProcessSystem`. Each "
                       "arrow is a stream labelled with its temperature (°C), pressure "
                       "(bara) and mass flow (kg/hr). Recycle loops close back on "
                       "their feed units. **Scroll to zoom, drag to pan.**")
            c1, c2 = st.columns([3, 1])
            diagram_height = c1.slider("Diagram height (px)", min_value=480,
                                       max_value=1400, value=760, step=40,
                                       key="teg_diagram_height")
            font_scale = c2.selectbox("Text size", ["Normal", "Large", "Extra large"],
                                      index=1, key="teg_diagram_fontscale")
            _fs = {"Normal": (16, 13, 11), "Large": (20, 16, 14),
                   "Extra large": (26, 22, 18)}[font_scale]
            dot_render = _enlarge_dot_fonts(dot_graph, _fs[0], _fs[1], _fs[2])
            try:
                render_graphviz_interactive(dot_render, height=int(diagram_height))
            except Exception:
                # Fallback to the static renderer if the HTML component fails
                st.graphviz_chart(dot_render, use_container_width=True)
            st.download_button(
                "⬇️ Download flow diagram (Graphviz DOT)",
                dot_graph.encode('utf-8'),
                file_name="teg_flow_diagram.dot", mime="text/vnd.graphviz",
            )

    # Full process-model results as JSON (every unit operation + emissions)
    process_report = res.get('process_report')
    if process_report:
        with st.expander("🧾 Full process model results (JSON — all unit operations)",
                         expanded=False):
            st.caption("Complete dump of the solved NeqSim `ProcessSystem`: every "
                       "unit operation with its result table, and each stream with its "
                       "temperature, pressure, flow and overall molar composition. The "
                       "atmospheric still-vent and flash-gas emission breakdowns are "
                       "included under `emissions`.")
            full_report = {
                'units': process_report.get('units', []),
                'emissions': {
                    'still_vent_kg_hr': still,
                    'flash_gas_kg_hr': flash,
                },
                'operating_point': {
                    'flash_drum_pressure_bara': flash_pressure_r,
                    'feed_pressure_bara': res['feed_pressure'],
                    'stripping_gas_recirculated': res['recirculate'],
                    'recycle_blower_kW': res['blower_kw'],
                    'water_dew_point_C': res['water_dew_C'],
                    'lean_teg_purity_wt_pct': res['lean_teg_wt'],
                    'dry_gas_flow_MSm3_day': res['dry_gas_flow'],
                },
            }
            st.json(full_report, expanded=False)
            st.download_button(
                "⬇️ Download full results (JSON)",
                json.dumps(full_report, indent=2).encode('utf-8'),
                file_name="teg_model_results.json", mime="application/json",
            )

    # Model input / builder JSON (round-trippable model definition)
    model_input_json = res.get('model_input_json')
    if model_input_json:
        with st.expander("🧩 Model input (JSON — convertible to a ProcessSystem)",
                         expanded=False):
            st.caption("Builder JSON for the TEG plant in the NeqSim "
                       "`JsonProcessBuilder` schema. This is the *input* definition "
                       "of the model (equipment, streams, fluids and connections), not "
                       "the solved results. It can be reloaded with "
                       "`ProcessSystem.fromJsonAndRun(json)` to rebuild and re-run the "
                       "identical process.")
            try:
                st.json(json.loads(model_input_json), expanded=False)
            except Exception:
                st.code(model_input_json, language="json")
            st.download_button(
                "⬇️ Download model input (JSON)",
                model_input_json.encode('utf-8'),
                file_name="teg_model_input.json", mime="application/json",
            )

    # KPIs
    st.subheader("Dehydration performance")
    k1, k2, k3 = st.columns(3)
    k1.metric("Dry-gas water dew point", f"{res['water_dew_C']:.2f} °C",
              help=f"At {res['feed_pressure']:.0f} bara reference")
    k2.metric("Lean TEG purity", f"{res['lean_teg_wt']:.2f} wt%")
    k3.metric("Dry gas flow", f"{res['dry_gas_flow']:.3f} MSm³/day")

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
        title=f'TEG regeneration vent emissions (flash drum {flash_pressure_r:.1f} bara)',
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
    if res['sweep']:
        st.divider()
        st.subheader("Flash-drum pressure sensitivity")
        sweep_pressures = res['sweep']['pressures']
        still_nmvoc = res['sweep']['still_nmvoc']
        flash_nmvoc = res['sweep']['flash_nmvoc']

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

        if 9.0 in sweep_pressures and 2.0 in sweep_pressures:
            hi = still_nmvoc[sweep_pressures.index(9.0)]
            lo = still_nmvoc[sweep_pressures.index(2.0)]
            if hi > 0:
                st.info(f"Lowering the flash drum from 9.0 → 2.0 bara reduces atmospheric "
                        f"still-vent NMVOC from {hi:.3f} to {lo:.3f} kg/hr "
                        f"(**{100.0*(hi-lo)/hi:.1f}% reduction**).")
else:
    st.info("Set the inputs above and press **Run emission calculation**.")

st.divider()
st.caption("Model: CPA EOS (SystemSrkCPAstatoil, mixing rule 10). NMVOC = non-methane "
           "volatile organic compounds. The still vent is the atmospheric emission; "
           "flash gas is normally recovered to fuel.")
