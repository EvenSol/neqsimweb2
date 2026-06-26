"""Standalone smoke test for the TEG dehydration emission page logic.

Mirrors build_teg_plant / classify_emissions from
neqsimweb2/pages/81_TEG_Dehydration_Emissions.py to verify the plant converges
and produces emission numbers with the dynamic (configurable) composition arrays.
"""
from neqsim import jneqsim

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

GAS_COMPONENTS = ['nitrogen', 'CO2', 'methane', 'ethane', 'propane', 'i-butane',
                  'n-butane', 'i-pentane', 'n-pentane', 'n-hexane', 'benzene']
DEFAULT_FEED = [0.245, 3.4, 85.7, 5.981, 2.743, 0.37, 0.77, 0.142, 0.166, 0.06, 0.01]
NMVOC = {'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane',
         'n-hexane', 'benzene'}
GHG_CH4 = {'methane'}


def build_teg_plant(feed_fractions, feed_flow_MSm3_day, feed_temp_C, feed_pressure_bara,
                    absorber_pressure_bara, absorber_temp_C, teg_flow_kg_hr, teg_feed_temp_C,
                    lean_teg_purity, flash_drum_pressure_bara, reboiler_temp_C,
                    stripping_gas_Sm3_hr, n_absorber_stages, stage_efficiency,
                    water_mode='saturated', water_content_ppm_mol=None,
                    saturation_temp_C=None, saturation_pressure_bara=None):
    p = ProcessSystem()
    n_comp = len(GAS_COMPONENTS) + 2

    feedGas = SystemSrkCPA()
    total_gas = sum(float(f) for f in feed_fractions)
    for name, frac in zip(GAS_COMPONENTS, feed_fractions):
        feedGas.addComponent(name, float(frac))
    if water_mode == 'specified':
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
        wetFeedGas = dryFeedGas
    elif saturation_temp_C is not None or saturation_pressure_bara is not None:
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

    feedTEG = feedGas.clone()
    leanComp = [0.0] * n_comp
    leanComp[-2] = 1.0 - lean_teg_purity
    leanComp[-1] = lean_teg_purity
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
    heatEx2.setUAvalue(2224.0)
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
    heatEx.setUAvalue(8316.0)
    p.add(heatEx)

    flashValve2 = ThrottlingValve('Rich TEG LP flash valve', heatEx.getOutStream(0))
    flashValve2.setOutletPressure(1.2)
    p.add(flashValve2)

    stripGas = feedGas.clone()
    strippingGas = Stream('stripGas', stripGas)
    strippingGas.setFlowRate(stripping_gas_Sm3_hr, 'Sm3/hr')
    strippingGas.setTemperature(78.3, 'C')
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
    column.getCondenser().setOutTemperature(273.15 + 85.0)
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
    stripper.setNumberOfStages(2)
    stripper.setStageEfficiency(1.0)
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
    makeupComp[-1] = 1.0
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
        'dehydratedGas': dehydratedGas, 'flashGas': flashGas, 'stillVent': stillVent,
        'leanTEGtoAbs': leanTEGtoAbs, 'waterDewAnalyser': waterDewAnalyser,
    }
    return p, streams


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
        'water': flows.get('water', 0.0),
        'TEG': flows.get('TEG', 0.0),
        'benzene': flows.get('benzene', 0.0),
    }
    out['total'] = sum(flows.values())
    return out


if __name__ == '__main__':
    kwargs = dict(
        feed_fractions=DEFAULT_FEED, feed_flow_MSm3_day=4.65, feed_temp_C=25.0,
        feed_pressure_bara=70.0, absorber_pressure_bara=85.0, absorber_temp_C=35.0,
        teg_flow_kg_hr=5500.0, teg_feed_temp_C=48.5, lean_teg_purity=0.97,
        flash_drum_pressure_bara=4.8, reboiler_temp_C=197.5, stripping_gas_Sm3_hr=180.0,
        n_absorber_stages=4, stage_efficiency=0.7,
    )
    print('Building plant...')
    process, S = build_teg_plant(**kwargs)
    print('Running (worker thread)...')
    thr = process.runAsThread()
    thr.join(300000)

    water_dew_C = float(S['waterDewAnalyser'].getMeasuredValue('C'))
    lean_teg_wt = teg_mass_fraction(S['leanTEGtoAbs'])
    dry_gas_flow = float(S['dehydratedGas'].getFlowRate('MSm3/day'))
    still = classify_emissions(S['stillVent'])
    flash = classify_emissions(S['flashGas'])

    print('\n=== RESULTS ===')
    print(f'Water dew point     : {water_dew_C:8.2f} C @70 bara')
    print(f'Lean TEG purity     : {lean_teg_wt:8.2f} wt%')
    print(f'Dry gas flow        : {dry_gas_flow:8.3f} MSm3/day')
    print(f'Still-vent NMVOC    : {still["NMVOC"]:8.4f} kg/hr')
    print(f'Still-vent methane  : {still["methane"]:8.4f} kg/hr')
    print(f'Still-vent CO2      : {still["CO2"]:8.4f} kg/hr')
    print(f'Still-vent water    : {still["water"]:8.4f} kg/hr')
    print(f'Still-vent benzene  : {still["benzene"]:8.5f} kg/hr')
    print(f'Flash-gas NMVOC     : {flash["NMVOC"]:8.4f} kg/hr')

    assert water_dew_C < 0.0, 'dew point should be below 0 C after dehydration'
    assert lean_teg_wt > 90.0, 'lean TEG should be >90 wt%'
    assert still['NMVOC'] >= 0.0 and flash['NMVOC'] >= 0.0
    print('\nSMOKE TEST (saturated) PASSED')

    # --- Specified water content mode ---
    print('\nBuilding plant (specified water content = 1000 mol-ppm)...')
    kwargs2 = dict(kwargs, water_mode='specified', water_content_ppm_mol=1000.0)
    process2, S2 = build_teg_plant(**kwargs2)
    print('Running (worker thread)...')
    thr2 = process2.runAsThread()
    thr2.join(300000)

    water_dew_C2 = float(S2['waterDewAnalyser'].getMeasuredValue('C'))
    lean_teg_wt2 = teg_mass_fraction(S2['leanTEGtoAbs'])
    still2 = classify_emissions(S2['stillVent'])
    print(f'Water dew point     : {water_dew_C2:8.2f} C @70 bara')
    print(f'Lean TEG purity     : {lean_teg_wt2:8.2f} wt%')
    print(f'Still-vent water    : {still2["water"]:8.4f} kg/hr')
    assert water_dew_C2 < 0.0, 'dew point should be below 0 C after dehydration'
    assert lean_teg_wt2 > 90.0, 'lean TEG should be >90 wt%'
    print('\nSMOKE TEST (specified) PASSED')
