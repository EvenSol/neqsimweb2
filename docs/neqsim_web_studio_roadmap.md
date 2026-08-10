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

Status: **professional results and design review in current root PR**.

The current tranche adds a shared, UI-independent exact-result handoff and a
professional results workspace over the existing solved model. It exposes:

- explicit solved/warning/attention state without presenting stale or dirty results;
- KPIs, units, thermodynamics, assumptions and solver provenance;
- streams, equipment, design capacities, margins, utilization and limit status;
- system/component/unit conservation and recycle/adjuster convergence evidence;
- display-safe session case comparison while retaining sensitivity, adjust and
  bounded optimization execution in the inherited flowsheet tools.

This is validated-but-unmerged stack evidence and is not counted as completed
until its root PR is merged.

### S5 — Engineering Drawings

Status: dependent on merged core capabilities.

Consume the canonical PFD/P&ID/DEXPI model from `equinor/neqsim`, including stable
semantic IDs, drawing sets/sheets, validation diagnostics and export artifacts.
The web layer owns interactive engineering workflow, not standards semantics.

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

Status: ongoing across all milestones.

Browser/end-to-end tests, accessibility, performance, safe execution isolation,
migration/interoperability evidence and deployment validation are release gates.

## Current tranche

S4 professional active-case results is the next root tranche after the S0–S3/S6
foundation merged to `main`. It reuses the exact `flowsheet_studio_case` solved
state, `NeqSimProcessModel`, shared solver-diagnostic adapters, equipment design
properties and solved case history. It adds no calculation engine and changes no
Classic route.

Acceptance requires fail-closed exact-signature validation, explicit units and
provenance, honest engineering status, focused result-projection regressions,
Classic + Studio application health and the existing native conservation gate.

## Active stacked work

The current S4 results/design tranche is a new root PR targeting `main`. There are
no child Studio implementation PRs above it. It is recorded as **in stack**, not
merged completion.

The next dependency-ready tranche is deeper studies integration through the same
shared case service, followed by a core-API-aligned Engineering Drawings adapter
when the #1332/#2899 contracts are merged and stable.
