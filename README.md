![NeqSim Logo](https://github.com/equinor/neqsim/blob/master/docs/wiki/neqsimlogocircleflatsmall.png)

# NeqSim Web application
The NeqSim Web project is a web user interface for the [neqsim process simulator](https://equinor.github.io/neqsimhome/).

The application is in initial development and is using the [streamlit framework](https://streamlit.io/).

The application is hosted by streamlit and can be tested at [neqsim web app link](https://neqsim.streamlit.app/).

## Pipeline Hydraulics

The **Pipeline** page provides two native NeqSim calculation paths:

- **Steady-State (Beggs-Brill)** uses `PipeBeggsAndBrills` for liquid
  holdup, flow regime, friction, hydrostatic pressure change, thermodynamics,
  and optional specified-U heat transfer. The app solves inlet pressure to
  match the requested outlet pressure and reports NeqSim's native increment
  profiles.
- **Dynamic Simulation (Two-Fluid Model)** uses `TwoFluidPipe` for separate
  gas and liquid conservation equations, terrain, heat transfer, flow-regime
  transitions, and optional slug tracking. A converged native steady state is
  used to initialize the transient calculation.

Both paths use the same normalized fluid table. The page starts with a lean
natural-gas preset and also provides a two-phase gas-condensate regression
fluid and a blank custom fluid. SRK is the explicit default thermodynamic
model; PR, CPA, and NeqSim automatic model selection are available in the
sidebar. Standard gas rate is interpreted at 1.01325 bara and 15 °C; actual
volume rate uses the entered pressure and temperature. Pressures are absolute
unless a result is explicitly labelled as a pressure drop.

Terrain distances must be strictly increasing. The dynamic page shifts the
first distance to zero and interpolates absolute elevations at the native
two-fluid section centres. Zero mass flow is rejected before the Java model is
called. Results are intended for engineering screening and require model
validity, discretization, and time-step sensitivity checks before project use.

## Process Flowsheet Studio

Process Flowsheet Studio builds and solves reusable steady-state NeqSim cases
from shared fluid packages, independent inlet streams, unit operations, and
explicit material or energy connections.

Enable an equipment design basis in a pump, two-sided heat exchanger, valve,
or pipeline to compare its solved operating point with explicit capacities.
The **Workbook · Design** view reports operating value, capacity, margin,
utilization, status, and engineering unit in a normalized table. Pipeline rows
also identify the critical native velocity-profile segment and length when the
NeqSim unit exposes that profile.

The downloadable engineering workbook contains the same review-ready data in
the **Equipment Design** worksheet together with streams, equipment,
constraints, convergence, and conservation evidence. These results support
screening and engineering studies; they are not design certification.

Schema v4 adds execution-neutral **subflowsheets** on top of the authoritative
flat process graph. A subflowsheet owns a non-overlapping set of units and
declares every material or energy port used where a connection crosses its
boundary. Terminal product ports may also be exposed explicitly. Studio
validates these contracts during import, draft history, execution planning,
and solve readiness; the draft diagram renders each group as a labeled dashed
container and lists its boundary-port mappings. Schema-v1–v3 cases migrate to
v4 with an empty subflowsheet list, so existing calculations remain unchanged.

Studio runs each native flowsheet build, convergence pass, mixer-energy
closure, design rerun, and final solve within one 180-second execution budget.
Workers that exceed the budget are interrupted or abandoned after a bounded
cancellation wait, and failed or timed-out native models are discarded instead
of being published as solved results. Timeout classification survives normal
Streamlit reruns while the last trustworthy result stays hidden.

The current UniSim-parity assessment and production-readiness gaps are tracked
in [docs/process_flowsheet_unisim_parity.md](docs/process_flowsheet_unisim_parity.md).

## NeqSim Games

**NeqSim Games** is the shared Studio menu for native calculation challenges
and reusable flashcard games.

**Plant Operator** is backed by native NeqSim process calculations. Its first
scenario, **The 10% Throughput Challenge**, starts from a synthetic 100,000
kg/hr gas-compression case. The player changes feed rate, interstage pressure,
export pressure, intercooling, and export cooling to reach at least 110,000
kg/hr without violating the published product, power, temperature, cooling, or
conservation limits. Build, solve, and evidence extraction share one
180-second execution budget; partial or timed-out models are not published.

**Phase Equilibrium Lab** runs a bounded native SRK TP flash for a fixed rich
gas. The player searches temperature and pressure for a narrow retrograde
window defined by phase split, gas density, gas Z-factor, liquid density,
liquid viscosity, and phase-fraction closure. The result includes gas and
liquid compositions, component K-values, mixture properties, and downloadable
evidence.

The Games hub also provides a built-in phase-equilibrium and fluid-properties
flashcard deck. Users can author their own cards, play the custom deck, track
session mastery, and import or export a bounded versioned JSON deck.

See [docs/neqsim_plant_operator_game.md](docs/neqsim_plant_operator_game.md)
for the process scenario contract. See
[docs/neqsim_games.md](docs/neqsim_games.md) for the shared menu, phase-game
targets, native evidence, flashcard schema, and validation boundaries.
