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

### S5 — Engineering Drawings

Status: **blocked on a merged, web-consumable core drawing-set contract**.

Current `equinor/neqsim` master includes the canonical topology-backed assessed
DEXPI 2.0 Process material projection through #2932 and the opt-in successful-run
operating-case snapshot through #2934. The latter adds stable case/object identity,
K, bara absolute and kg/s values, provenance and fail-closed unsuccessful-run
diagnostics. It is reusable core foundation, but the assessed DEXPI writer does
not yet consume those values and this is not a controlled Studio drawing set.
Document/sheet identity, revision/status, layout ownership, off-page references
and qualified SVG/PDF artifacts remain future core work. Studio therefore keeps
Engineering Drawings honestly marked as core integration in progress rather than
defining a competing payload or renderer.

The exact activation boundary and adapter responsibilities are recorded in
`docs/neqsim_web_studio_drawing_contract.md`.

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

Status: dependent on validated NeqSim core handoff/dynamics.

### S8 — Large ProcessModel / multi-area workspace

Status: planned.

### S9 — Production hardening

Status: **ongoing; central-route health coverage in the active stack**.

Browser/end-to-end tests, accessibility, performance, safe execution isolation,
migration/interoperability evidence and deployment validation are release gates.
The active S9 child tranche adds the central Process Flowsheet page to the fresh
Streamlit root/health/live-process gate and requires every enabled Studio
destination to remain represented in that gate. This is in-stack evidence, not a
merged production-readiness claim.

The top S9 accessibility tranche gives each disabled roadmap workflow a unique
native button name. Screen-reader and keyboard users can distinguish which
Thermodynamics, Dynamics, Engineering Drawings or Examples action is unavailable
without changing its disabled state or overstating capability. This is also
in-stack evidence.

## Current tranche

Recovery PR #100 merged at `90e1477ca7761c8bfe288f8b6f302380ab49aada`,
so exact-model sensitivity, bounded optimization, scenario, emissions and
energy-audit evidence is now on `main`. The merge preserves Classic routes and
portable schema behavior, and the complete hosted gate passed Classic, Studio,
Results, Process Chat and pipeline health plus warm deployment, native
conservation, bounded execution and pipeline hydraulics.

The root documentation tranche reconciles the roadmap with that merge and freezes
the Engineering Drawings integration boundary at verified merged core evidence.
It records merged core PR #2934 without activating a runtime adapter because the
remaining document, sheet and artifact contracts are still absent. Its S9 children
harden deployment coverage for the inherited central Process Flowsheet workflow
and distinguish disabled dashboard actions without changing calculations, schema,
routes or saved cases.

## Active stacked work

PR #100 is merged. The active stack contains the documentation-only drawing
contract root targeting `main`, its S9 deployment-health child and a top S9
dashboard-accessibility child. Each child targets its immediate parent's exact
validated head. No tranche claims that unavailable `equinor/neqsim` capability is
available.

While the drawing-set contract remains unavailable, further safe S9 work is
Classic/Studio browser interaction, accessibility and migration/interoperability
hardening. Engineering Drawings can move into an interactive Studio adapter only
after the remaining core contracts are merged and available in the NeqSim runtime
used by the web application.
