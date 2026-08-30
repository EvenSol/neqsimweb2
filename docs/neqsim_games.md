# NeqSim Games

NeqSim Games is the shared Studio entry point for engineering simulation games
and reusable flashcard games. Open **NeqSim Studio → NeqSim Games** to reach:

- **Plant Operator** — a native process-simulation optimization challenge;
- **Phase Equilibrium Lab** — a native thermodynamics and fluid-property
  challenge; and
- **Flashcard game** — a built-in learning deck plus a session-local deck
  builder with JSON import and export.

The calculated games are educational. They expose their native evidence and do
not replace fluid characterization, equipment design, operating procedures, or
engineering approval.

## Phase Equilibrium Lab

### Challenge: Catch the Retrograde Window

The player selects temperature from -20 to 80 °C and absolute pressure from 10
to 130 bara for a fixed synthetic rich gas. Each attempt creates a new NeqSim
`SystemSrkEos`, applies mixing rule 2, enables multiphase checking, and executes
one bounded TP flash. The feed mole fractions are:

| Component | Mole fraction |
|---|---:|
| CO2 | 0.005 |
| methane | 0.720 |
| ethane | 0.080 |
| propane | 0.060 |
| i-butane | 0.025 |
| n-butane | 0.035 |
| i-pentane | 0.015 |
| n-pentane | 0.015 |
| n-hexane | 0.020 |
| n-heptane | 0.015 |
| n-octane | 0.010 |

One attempt wins only when every native evidence check passes:

| Check | Requirement | Score weight |
|---|---:|---:|
| Equilibrium phases | exactly gas + hydrocarbon liquid | 180 |
| Condensate split | 16–20 mol% | 220 |
| Gas density | 78–92 kg/m³ | 140 |
| Gas compressibility | Z = 0.80–0.83 | 140 |
| Liquid density | 480–510 kg/m³ | 120 |
| Liquid viscosity | < 0.12 cP | 100 |
| Phase-fraction closure | error ≤ 1×10⁻¹⁰ | 100 |

The maximum score is 1,000. A missing or non-finite native property fails its
check. A partial score cannot override a failed win condition.

The challenge was calibrated with a native grid sweep. A representative winning
point is 50 °C and 80 bara: approximately 16.93 mol% hydrocarbon liquid,
79.45 kg/m³ gas density, gas Z of 0.8186, 499.52 kg/m³ liquid density, and
0.1051 cP liquid viscosity. The starting point at 20 °C and 50 bara fails the
gas-density, Z-factor, liquid-density, and viscosity bands. This gives the game
a feasible but constrained operating region rather than a single hard-coded
answer.

### Equilibrium evidence

The result view reports phase fractions, densities, viscosities, gas Z,
mixture enthalpy, mixture Cp, and phase-fraction closure. It also reports feed,
gas, and liquid mole fractions for each component plus `K = y/x`. Values above
one identify components that favor the gas phase at that equilibrium point;
values below one identify components that favor the liquid phase.

The evidence export is version-neutral JSON containing the controls, solved
properties, component splits, K-values, assessment, and elapsed flash time.

## Flashcard games

The Games hub includes a built-in **Phase equilibrium & fluid properties** deck
covering TP flashes, dew and bubble points, K-values, retrograde condensation,
Z-factor, density, viscosity, heat capacity, equations of state, and evidence
validation. The player reveals each answer and self-marks it as mastered or for
review. Progress and mastery are session-local.

The deck builder accepts a topic, front, back, optional explanation, and one of
three difficulty levels: `foundation`, `applied`, or `advanced`. Custom cards
can be played immediately, cleared, downloaded, and imported later.

Portable deck JSON uses schema version 1:

```json
{
  "schema_version": 1,
  "name": "My custom NeqSim deck",
  "description": "A user-authored learning game.",
  "cards": [
    {
      "card_id": "custom-1",
      "topic": "Phase equilibrium",
      "prompt": "What does K > 1 mean?",
      "answer": "The component favors the gas phase.",
      "explanation": "K = y/x.",
      "difficulty": "applied"
    }
  ]
}
```

Imports are bounded to 100 cards and 100 kB, require unique card identifiers,
reject unknown schema versions, and validate all required text fields. Custom
prompt text is escaped before being placed in the styled card surface.

## Validation and execution boundaries

- Initial Games and Phase Lab page loads do not start NeqSim.
- A native flash starts only after the player presses the run button.
- Each flash has one 60-second caller wait budget; a timeout discards the
  partial fluid state.
- Player controls are finite and bounded before the JVM starts.
- Previous results become stale immediately when the controls change.
- Native property absence never becomes a passing check.
- Unit, initial-render, HTTP-health, native calibration, compilation, and
  navigation tests are part of the Process Flowsheet Studio workflow.

See [neqsim_plant_operator_game.md](neqsim_plant_operator_game.md) for the Plant
Operator process challenge contract.
