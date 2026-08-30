# NeqSim Plant Operator

NeqSim Plant Operator turns native process-simulation results into bounded,
auditable engineering challenges. It is an educational workflow for learning
how operating choices affect production, product conditions, energy use, and
process constraints. It is not design certification or operator-procedure
authorization.

## Challenge 1: The 10% Throughput Challenge

The supplied process is a synthetic gas-rich feed at 50 bara and 30 °C. The
fixed train is:

1. inlet scrubber;
2. first-stage compressor;
3. intercooler;
4. interstage scrubber;
5. second-stage compressor; and
6. export cooler.

The thermodynamic model is SRK with mixing rule 2. The two compressor
isentropic efficiencies are fixed at 0.78. The player controls feed mass flow,
the two compressor discharge pressures, intercooler outlet temperature, and
export-cooler outlet temperature. Pressures are absolute and all game inputs
use the units displayed in the interface.

### Win conditions

One attempt wins only when every check passes:

| Check | Requirement |
|---|---:|
| Feed throughput | at least 110,000 kg/hr |
| Export pressure | at least 128 bara |
| Export temperature | at most 45 °C |
| Maximum compressor discharge temperature | at most 120 °C |
| Total compressor power | less than 4,200 kW |
| Specific compression energy | at most 41 kWh/tonne |
| Total cooling duty | less than 5,500 kW |
| System mass-balance error | at most 0.10% |
| System energy-balance error | at most 0.10% |
| Other native NeqSim constraints | no failed or unavailable checks |

All values are extracted after the native NeqSim process has converged. The
game does not convert a missing, failed, or unavailable conservation result
into a pass.

## Score

The score is bounded from 0 to 1,000 points:

- production target: 400 points;
- stretch throughput above the 10% target: up to 50 points;
- specific compression energy: up to 250 points;
- compressor discharge temperature: up to 100 points;
- cooling-system load: up to 50 points; and
- mass balance, energy balance, and native validation integrity: 150 points.

The published win conditions are independent of the score. A high score cannot
override a failed engineering check. The score rewards efficient operation
inside the feasible region; the check table remains the authoritative outcome.

## Reproducibility and interoperability

Each attempt follows this chain:

`controls -> ProcessBuilder specification -> native NeqSim solve -> engineering evidence -> score`

Build, solve, and evidence extraction use one shared 180-second deadline. A
timeout discards the partial model. The interface hides a previous result when
the player changes a control, preventing stale evidence from being presented
as current.

The solved ProcessBuilder specification can be downloaded as JSON. The exact
in-memory solved `NeqSimProcessModel` can also be handed to Process Chat within
the same Streamlit session.

## Current scope and next increments

Version 1 is a steady-state, single-scenario training game. It does not yet
include dynamic controllers, compressor trips, hydrate-margin calculations,
economic prices, emissions pricing, persistent leaderboards, or multiplayer
state. Those should be added as separate validated challenges so that each new
game mechanic retains a clear NeqSim calculation and acceptance contract.
