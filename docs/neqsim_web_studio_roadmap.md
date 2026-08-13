# NeqSim Web Studio roadmap

## Product contract

NeqSim Web is evolving into two user-selectable workspaces in the same application:

- **NeqSim Classic** preserves the current pages, calculations, saved cases and familiar sidebar-driven workflow.
- **NeqSim Studio (Beta)** is the new case-based professional engineering workspace.

Studio does not introduce a second simulation engine. `equinor/neqsim` remains the
authoritative calculation and engineering core. `EvenSol/NeqSim-Colab` supplies
executable reference workflows and acceptance evidence where a Studio workflow
needs a reproducible engineering example.

## Inherited Process Flowsheet Studio foundation

The new Studio campaign inherits the merged Process Flowsheet Studio roadmap rather
than restarting it. Current evidence includes:

- shared fluid packages and independent inlet conditions;
- explicit named material/energy connections and generic graph execution;
- mixers, splitters, phase outlets, scalable multi-inlet topology and subflowsheets;
- searchable graph editing, property metadata, history and unsolved-draft persistence;
- convergence plus mass/component/energy closure diagnostics;
- native `.neqsim` persistence, schema migration, workbook/export and Process Chat handoff;
- sensitivity/adjust/optimization tools and equipment-design screening/datasheets;
- timeout/fail-loud execution, warm-deployment regressions and Streamlit health gates.

`docs/process_flowsheet_unisim_parity.md` remains the detailed capability/gap
assessment for that inherited foundation.

## Repository responsibilities

| Repository | Studio role |
| --- | --- |
| `equinor/neqsim` | Authoritative thermodynamics, process simulation, equipment/design, dynamics, optimization and canonical PFD/P&ID/DEXPI semantics |
| `EvenSol/neqsimweb2` | Classic + Studio UI, case orchestration, shared adapters/services, result presentation and workflow integration |
| `EvenSol/NeqSim-Colab` | Executable reference workflows, acceptance cases, teaching and engineering validation evidence |

The ISO PFD campaign (#1332) and DEXPI/P&ID campaign (#2899) remain owned by
`equinor/neqsim`. Studio consumes merged APIs and must not create a competing
diagram engine.

## Milestones

### S0 — Classic preservation and architecture baseline

Status: **completed foundation on `main`**.

- Classic homepage content and existing page navigation remain usable.
- Studio is introduced side-by-side rather than replacing existing pages.
- Shared services are extracted only when behavior is preserved by regression tests.

Merged evidence: Studio shell stack PR #93 and its Classic/Studio navigation and
health regressions.

### S1 — Studio shell and workspace choice

Status: **completed foundation on `main`**.

- A professional Studio dashboard is available as a separate Streamlit page.
- Users can move explicitly between Studio and Classic.
- The mature Process Flowsheet Studio is the first enabled Studio engineering workflow.
- Planned workflows remain visible with honest availability state.

Merged evidence: Studio shell stack PR #93.

### S2 — Shared case context and lifecycle

Status: **completed foundation on `main`**.

Create/open/save/save-as/reset, metadata, units, thermodynamic package, provenance
and solved/dirty/error state are shared across Studio pages without corrupting
Classic cases.

Merged evidence from Studio stack PR #94:

- UI-independent `studio.case_context` owns only Studio session lifecycle metadata;
- schema-v1–v4 Process Flowsheet Studio JSON remains the authoritative portable case contract;
- New, Open, Download, Save As, Reset and session-local Recent Cases are available;
- stable identity, explicit units, thermodynamic summary, provenance and lifecycle state span Studio pages;
- native model availability/signature are runtime evidence while Java/Python objects remain in the established adapter;
- lifecycle regressions prove unrelated Classic session data is not altered.

### S3 — Integrate mature Process Flowsheet Studio

Status: **initial integration completed on `main`; broader workspace integration continues**.

The active Studio case hands its established solved `process_model` directly to
Process Chat. The generic editor, builder, diagnostics, workbooks, studies and
direct Classic-compatible flowsheet route remain the inherited implementation.

Merged evidence: Studio stack PRs #94 and #95.

### S4 — Professional results, design and engineering studies

Status: **completed professional results, design and study-evidence tranche on `main`**.

The merged tranche adds a shared, UI-independent exact-result handoff and a
professional results workspace over the existing solved model. It exposes:

- explicit solved/warning/attention state without presenting stale or dirty results;
- KPIs, units, thermodynamics, assumptions and solver provenance;
- streams, equipment, design capacities, margins, utilization and limit status;
- system/component/unit conservation and recycle/adjuster convergence evidence;
- display-safe session case comparison while retaining sensitivity, adjust and
  bounded optimization execution in the inherited flowsheet tools.

Professional results and design review are merged through PR #96. Recovery PR
#100 placed the reviewed exact-model engineering-study evidence from closed child
PRs #98 and #99 onto current `main`. Completed Process Chat sensitivity sweeps,
bounded optimization, scenario comparisons, emissions and energy-audit evidence
are projected only when the chat session references the same native model and
solved signature. Failed points/scenarios, units, method, constraints, bottlenecks,
source-level emissions, utility consumers and screening limitations remain visible.

Study execution stays in the existing NeqSim-backed tools. Mutation-capable
bounded optimization runs on a fail-closed isolated model clone so auto-sizing,
chart setup and search reruns cannot alter the active solved Studio case. Studio
does not rerun or duplicate the calculations.

Merged core #2964 / `a0a011f2` adds explicit constraint scaling and conservative
candidate-active diagnostics. Merged core #2975 / `7334454c` adds reversible,
identity- and provenance-bearing continuous/discrete operating actions with
strict candidate validation, write/read-back verification and deterministic
restoration. These APIs are future shared-service inputs; they do not replace the
current isolated-clone safety boundary until a deployed NeqSim runtime exposes
them and exact-model Studio regressions prove equivalent restoration, feasibility
and evidence behavior.

### S5 — Engineering Drawings

Status: **core document semantics merged; Studio activation remains blocked on
runtime and artifact contracts**.

Current `equinor/neqsim` master includes the canonical topology-backed assessed
DEXPI 2.0 Process material projection through #2932, the opt-in successful-run
operating-case snapshot through #2934, and canonical-value DEXPI consumption
through merged #2938 / `810415b0`. The five-argument assessed writer binds finite,
case-matched calculation nodes by stable canonical owner identity, converts K to
degree Celsius and kg/s to kg/h, preserves absolute bara, and diagnoses omitted
values without a live-stream fallback.

Merged core #2960 / `e8991c41` adds deterministic simple, branched and multi-area
drawing reference cases. Merged core #2961 / `74022651` adds the immutable
`EngineeringDiagramDocumentSet`, drawing-register and sheet identity,
revision/status/issue-purpose metadata, stable semantic and connector IDs,
reciprocal off-page references, structured diagnostics, deterministic
JSON/fingerprints, and `ProcessSystem`/`ProcessModel` adapters. This resolves the
controlled semantic document/sheet dependency without moving engineering meaning
into the web layer.

Merged core #2966 / `c8061c9d` now adds immutable governed semantic-object
snapshots for single- and multi-area operating cases. It retains stable source
identity and designations, case-scoped calculated values, explicit units,
quantity basis, engineering/approval state and provenance. Source names are not
silently promoted to approved equipment tags or line numbers.

Studio remains disabled because the deployed NeqSim Web runtime must expose the
#2961/#2966 APIs and the core still deliberately excludes layout/routing, symbols
and title blocks, manual overrides, qualified SVG/PDF artifacts, and a native
DEXPI document/graphics projection. Studio must also prove exact active
solved-signature binding and fail closed on mismatched snapshots. The exact
activation boundary and adapter responsibilities are recorded in
`docs/neqsim_web_studio_drawing_contract.md`.

NeqSim-Colab PR #119 is merged on `master` as canonical executable acceptance
evidence for the assessed DEXPI Process/PFD projection. It validates the existing
simulation-to-DEXPI path, but it does not supply the remaining runtime and
artifact contracts or the Studio-side exact-solution binding, so it does not
activate a Studio drawing adapter.

### S6 — Process Chat engineering copilot

Status: **initial case-aware integration completed on `main`**.

Merged Studio stack PR #95 exposes Process Chat as an available Studio workflow,
identifies the active case and thermodynamic/lifecycle state, and reuses the
existing solved-model handoff. It also supplies a bounded whitelisted case
projection that excludes the portable case body and arbitrary session state,
treats user-authored text as untrusted data and never promotes dirty/failed state
to solved evidence. Numeric answers remain sourced from the live model or an
executed deterministic NeqSim calculation.

### S7 — Dynamics and controls

Status: **core transaction foundation merged; Studio activation remains blocked
on complete equipment coverage and qualified adaptive stepping**.

Merged core #2965 / `fb206f20` adds opt-in identity-preserving
`ProcessSystem`/`ProcessModel` step transactions, quantitative fail-closed
coverage diagnostics, rollback/replay and multi-area atomicity. Merged core #2969
/ `ad66814e` adds the first built-in state-family adoption for base PID
controllers.

This is rollback architecture, not a validated dynamic workflow. Built-in
equipment, instruments, controller subclasses, recycles and external side
effects remain deliberately incomplete; full-step/two-half-step error estimation,
rejected-step retry, conservation/timestep evidence and OTS or safety
qualification are not established.

Merged NeqSim-Colab #121 / `df7186c2` adds bounded executable acceptance evidence
for a native SRK-CPA gas-oil-water `TwoFluidPipe` shutdown cooldown and OpenFOAM
dead-leg boundary handoff. Its synthetic 12 km case retained 14.14 h axial
no-touch time, 0.078% timestep difference, relative mass drift below
`3.1e-15` and thermal-energy residual below `1.3e-12`. This is a flow-assurance
reference, not long-duration compositional, hydrate-kinetics/pluggage, CFD,
general dynamic-process or controls qualification.

Studio Dynamics therefore remains disabled until a deployed runtime exposes a
representative fully covered process and the web handoff can preserve exact case
identity, accepted-step evidence and restart state without implementing dynamics
in Python.

### S8 — Large ProcessModel / multi-area workspace

Status: planned.

### S9 — Production hardening

Status: **ongoing; interoperability evidence merged, cross-page gate in stack**.

Browser/end-to-end tests, accessibility, performance, safe execution isolation,
migration/interoperability evidence and deployment validation are release gates.
Merged recovery PR #108 restores the central Process Flowsheet fresh-process
health gate, hosted coverage for every enabled Studio destination, unique native
action names and independent Recent Cases actions.

Root PR #109 added browser-level migration evidence through the real shared
Studio-to-flowsheet handoff. It covers schema v1-v4 canonicalization and
round-trip export, new identity for external opens, Classic session isolation and
fail-closed rejection of unsupported future schema v5. It is merged on `main` at
`9e2b1d98f13d4d6df7cb75314a24a9da0a5b293d`.

Root PR #110 is the current **in stack** S9 tranche. Its hosted Streamlit
interaction gate traverses the real Classic entrypoint, Studio Home, inherited
Process Flowsheet Studio and Engineering Results pages. It requires stable active
case identity and untouched unrelated Classic session state across New, Continue,
Recent Case, Equipment Design, Engineering Studies and fail-closed unsolved-result
actions.

## Current tranche

Recovery PR #100 merged at `90e1477ca7761c8bfe288f8b6f302380ab49aada`,
so exact-model sensitivity, bounded optimization, scenario, emissions and
energy-audit evidence is now on `main`. The merge preserves Classic routes and
portable schema behavior, and the complete hosted gate passed Classic, Studio,
Results, Process Chat and pipeline health plus warm deployment, native
conservation, bounded execution and pipeline hydraulics.

Documentation root PR #101 merged at
`d6a568d9c908a737830643fa4bc341a1c061119c` and froze the Engineering Drawings
activation boundary. Child PRs #106 and #107 were then merged into already-merged
parent branches, so their validated health and accessibility payloads did not
reach `main`.

Recovery PR #108 merged at
`073e90f56df6b20a5fadaf3b3c86b9ff7448c01b`, placing those exact
application/test payloads on `main` without changing calculations, schemas,
saved cases, Classic pages or engineering claims.

Root PR #109 merged the migration/interoperability gate: supported schema v1-v4
cases traverse the actual shared pending-case contract and existing flowsheet
importer, become canonical schema v4 and round-trip through the shared portable
serializer.
Unsupported future schema v5 must leave the active Studio case, Recent Cases and
unrelated Classic state unchanged.

Root PR #110 now advances browser interaction coverage without changing
production calculations or case schemas. The test uses the actual multipage
Streamlit entrypoint and existing shared services, including the unsolved Results
guard that returns engineers to the active flowsheet instead of presenting stale
evidence.

## Active stacked work

PR #109 is merged. Root PR #110 targets `main` from exact base
`9e2b1d98f13d4d6df7cb75314a24a9da0a5b293d` and is the only active Studio
implementation PR. Its cross-page interaction assertions are **in stack**, not
merged completion.

After #110 is validated and merged, the next safe S9 tranche is browser
interaction for uploaded supported/rejected portable case payloads and
deployment-level recovery.
Engineering Drawings can move into an interactive Studio adapter only after
controlled drawing-set and artifact contracts are merged and the required NeqSim
API is available in the web runtime.
