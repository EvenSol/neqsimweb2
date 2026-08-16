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
restoration. Merged core #2981 / `829def5e` adds defensive reversible hydraulic
action evaluation with explicit read-back resolution and tolerance evidence. These
APIs are future shared-service inputs; they do not replace the current
isolated-clone safety boundary until a deployed NeqSim runtime exposes them and
exact-model Studio regressions prove equivalent restoration, feasibility and
evidence behavior.

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
silently promoted to approved equipment tags or line numbers. Merged core #2979 /
`9eb9a42f` adds reviewed equipment/stream designations and deterministic semantic
revision impact while preserving those governance limits. Merged core #2987 /
`5d4dfe3e` adds opt-in project-controlled manual sheet definitions, stable
object assignments, pinned coordinates and protected routes with explicit review
evidence. These records preserve topology and do not imply engineering approval.

Merged core #3020 / `2e8a6fd4` adds an immutable governed stream-table
companion with stable stream identity, area and reviewed stream-number evidence,
explicit units and quantity bases, source calculation identity, provenance and
structured missing/duplicate/non-finite diagnostics. Merged core #3022 /
`e85d196b` stabilizes scale-aware deterministic DEXPI numeric serialization.
These strengthen future Studio drawing adapters without supplying the deployed
runtime, qualified artifact or exact-solution binding still required for
activation.

Studio remains disabled because the deployed NeqSim Web runtime must expose the
#2961/#2966/#2987 APIs and the core still deliberately excludes automatic
geometry/routing, symbols, legends and title-block geometry, qualified SVG/PDF
artifacts, and a native DEXPI document/graphics projection. Studio must also prove exact active
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
controllers. Merged core #2985 / `114209dd` extends transaction coverage to local
temperature and differential-pressure transmitters. Merged core #3019 /
`28025ff4` adds transactional fire-and-gas detector events, while #3021 /
`647b4b84` adds stateful pH-probe, soft-sensor and vibration-analyser
transaction coverage.

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

Status: **large-workspace projection, page-profile and Chromium baselines merged on `main`**.

Reviewed PR #115 established deterministic, UI-independent scale guards around
the inherited graph preview and shared professional-result projections. The
acceptance case contains 500 connected units, 2,000 solved stream rows, 1,000
equipment rows and 8,000 design-review rows. Coarse three-second hosted budgets
detect order-of-magnitude regressions without claiming production capacity,
browser rendering performance or NeqSim solver speed. Its merge commit
`45c5246ebb5e9fd649aa0e563a991c66b1f0221d` landed in the already-merged
#114 branch and therefore did not advance `main`.

Reviewed PR #116 runs that same solved-result scale through the actual Studio
Engineering Results page with Streamlit's browser-facing application harness. It
requires complete stream, equipment, design and constraint table payloads and
times full Streams and Equipment & design page reruns against coarse ten-second
hosted budgets. This isolates page/presentation overhead while leaving native
solving and production semantics unchanged. It is not a JavaScript paint,
network, memory, concurrent-user or NeqSim solver benchmark. Its merge commit
`27b03ef5f086fe1d117a6fea5cddb8560f7c13d8` landed in the already-merged
#115 branch and therefore did not advance `main`.

Recovery PR #117 subsequently merged on `main` at
`bdb0726ed984fd7a1bd6f87cfb7061762191e6ee`, restoring the cumulative
#115/#116 performance-test and workflow payload. Classic routes, calculations,
case schemas and saved-case behavior remain unchanged.

Validated child PR #118 adds a true Chromium paint/network/memory profile around
the same deterministic large solved workspace. Its exact-head hosted run recorded
first paint at 920 ms and first contentful paint at 992 ms; the complete Streams
view became ready in 2.546 s with 5,219,349 transferred bytes, 15,572,896 bytes
of JavaScript heap and 442 DOM nodes, while Equipment & design became ready in
0.282 s with 2,474,776 transferred bytes, 23,398,244 bytes of JavaScript heap
and 575 DOM nodes. These are reproducible regression observations, not production
capacity or concurrency claims. PR #118 merge commit
`97c40a9f4b51d71068bbb892d63d5264bac7f959` landed in the already-merged #117
branch and therefore did not advance `main`.

Recovery PR #119 merged on `main` at
`310dbcfa02ee1d31556657dfc5112af50342dbd3`, restoring the exact Chromium
profile payload. Its fresh hosted run recorded first contentful paint at 1.208 s,
Streams ready in 3.053 s and Equipment & design ready in 0.287 s, with network,
JavaScript heap, DOM and retained JSON artifact evidence. No repeatable evidence
currently justifies pagination, virtualization or caching.

### S9 — Production hardening

Status: **ongoing; interoperability merged, concurrent-solve recovery in stack**.

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

Root PR #110 merged at
`8a43e813fa794bba300a2d3b3b59e0cc3ccc8b39`. Its hosted Streamlit
interaction gate traverses the real Classic entrypoint, Studio Home, inherited
Process Flowsheet Studio and Engineering Results pages. It requires stable active
case identity and untouched unrelated Classic session state across New, Continue,
Recent Case, Equipment Design, Engineering Studies and fail-closed unsolved-result
actions.

PR #120 merged on `main` at
`3fe3bfa50eda6b7352c5a13dabcb6ee060e623c5`, adding the real Classic and
Studio Chromium journey at desktop and 390 × 844 mobile viewports. Recovery PR
#121 merged at `0ddb9acd78096e276b1bfb345e39924caa8b88f1` after exact-head run
#445 passed the complete Classic+Studio gate, inherited large-workspace profile
and real workspace journey. Four root/health probes stayed HTTP 200/`ok` with a
live process, desktop/mobile overflow was zero, and representative mobile targets
were 48.39 px high. Both evidence artifacts were retained. No production UI or
calculation behavior changed.

PR #122 merged on `main` at
`f0fb66b4a8d334b5d204cd05f61c2944f07aa74d`. Exact-head hosted run #447
passed two simultaneously live Chromium contexts: malformed portable case JSON
failed closed in one session while the peer retained a clean Studio state and
returned to Classic. Four root/health probes remained HTTP 200/`ok` with a live
process. No production code, calculation, schema or Classic behavior changed.

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

Root PR #110 merged browser interaction coverage without changing production
calculations or case schemas. The test uses the actual multipage Streamlit
entrypoint and existing shared services, including the unsolved Results guard that
returns engineers to the active flowsheet instead of presenting stale evidence.

Root PR #111 merged at
`de6c6a81249befb2e902d20f17f8b6375e1e23c8`. It exercises the existing
Studio upload action through the real multipage app: supported schema-v1 payloads
must migrate through the established flowsheet importer to canonical schema v4
under a new case identity. Invalid JSON and unsupported future-schema payloads
must fail closed without replacing the active case, Recent Cases or unrelated
Classic session state.

Root PR #112 merged on `main` at
`7b7786317f164254dd324b5d3c6533e574f3be59`.
It requires both rejected upload classes to recover through a subsequent supported
schema-v1 retry, keep the baseline and recovered cases in bounded Recent Cases,
consume one-shot pending state, preserve Classic session data and survive a normal
post-recovery Streamlit rerun.

Child PR #113 was merged into an already-merged parent branch and therefore did
not advance `main`. Recovery PR #114 subsequently merged on `main` at
`560e39187f6eff5947ec33387f6df1b44a60eb58`. It restores the exact #113
semantic section, article and heading structure, labelled status and active-case
regions, visible unique lifecycle action names, disabled planned states and owned
mobile layout contract without changing calculation or lifecycle behavior.

## Active stacked work

Recovery PR #119 merged on `main` at
`310dbcfa02ee1d31556657dfc5112af50342dbd3`, so the large-workspace
projection, page-rerun and Chromium profile baselines are now present together on
the default branch.

PR #122 merged on `main` at
`f0fb66b4a8d334b5d204cd05f61c2944f07aa74d`, so responsive accessibility,
transition recovery and session-isolated failure handling are now cumulative
default-branch evidence.

PR #123 merged on `main` at
`628a645afd8e9778f9ead1c9ac19272b267142a1`. Exact-head
`85f0337920c35e9ca8de34d806fb63970025e683` passed hosted run #456,
placing the retained full Chromium pilot on the default branch. The pilot traverses
Classic, Studio, the inherited starter-case native NeqSim solve, portable
schema-v4 JSON and 21-sheet engineering-workbook downloads, the existing live
solved-model Process Chat handoff without a provider call, and return to the same
solved flowsheet. It recorded zero page errors and four healthy root/application
probes with a live process.

PR #124 merged on `main` at
`98b002c3b78c07dc5effd3792f078cf8a0e59ecc`. Exact head
`982908b0b018eb4dd69a505ec2206c5e983c7363` passed hosted run #462,
placing the real-Chromium portable-case matrix and interoperability contract on
the default branch. Schemas v1-v4 canonicalized to v4 with UTF-8 BOM and Unicode
preservation, Classic continuation and fresh-session reopen equality. Malformed
JSON, non-UTF-8, 1,000,001-byte and future-v5 inputs failed closed and each
accepted a supported retry. Chromium 151 recorded zero page errors and four HTTP
200/`ok` probes with a live process.

Child PR #125 was merged into the already-merged #124 branch and therefore did
not advance `main`. Its first hosted run #463 passed the complete non-browser
Studio gate, large-workspace profile, accessibility journey and inherited full
native pilot. The new concurrent browser step exposed a Playwright/Streamlit
observation race: the success element repeatedly resolved visible but detached
during rerenders until a single stable wait timed out. No production calculation
or existing regression failed.

Recovery PR #126 targets current `main` from exact base
`98b002c3b78c07dc5effd3792f078cf8a0e59ecc`. It records the exact
process-isolation activation contract and the empirical stop boundary. Exploratory
two-session Chromium runs dispatched distinct native starter solves before either
completed and required independent JSON/workbook exports, solved Process Chat
handoffs, peer Classic navigation, four healthy probes and zero page errors.
Exact head `d8e7593a84ab06945501339fafe879c14409e5e3` passed run
31954372322 with requests 0.813 seconds apart and solve completion in 8.968 and
8.199 seconds. An unchanged documentation-only rerun then lost one session beyond
180 seconds, and a bounded process-local serialization experiment also failed the
repeatability gate. The experiment, browser gate and workflow wiring were removed;
this PR retains documentation only and changes no production or Classic behavior.
Reliable multi-session native execution remains **blocked** on the process worker
and exact solved-state rehydration contract below.
The exact isolated-worker dependency is recorded in
`docs/neqsim_web_studio_execution_isolation.md`. At current NeqSim master
`2e818a27d1fdd90017de41129ccdcd708bf49920`, the native
`ProcessSystem.loadFromNeqsim` and `ProcessModel.loadFromNeqsim` paths rerun
the deserialized model, while lifecycle JSON restoration still does not rebuild
equipment. Merged PR #3056 adds identity-preserving transient checkpoints for
base equipment state and configured `Recycle` units, but those checkpoints
restore selected state into an existing object graph and fail closed for uncovered
equipment; they do not reconstruct an arbitrary solved process from a worker
artifact. Studio therefore still cannot rehydrate an exact solved worker result
without a hidden second solve or incomplete process. Production subprocess
activation remains blocked until a merged no-run exact-state or equivalent
authoritative artifact contract is available in the deployed runtime. The current
bounded thread waits must not be described as process isolation.

Pagination, virtualization or caching remains gated on a repeatable user-visible
bottleneck.
Engineering Drawings can move into an interactive Studio adapter only after
controlled drawing-set and artifact contracts are merged and the required NeqSim
API is available in the web runtime.
