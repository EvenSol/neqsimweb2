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

Status: **in progress**

Acceptance:

- Classic homepage content and existing page navigation remain usable.
- Studio is introduced side-by-side rather than replacing existing pages.
- Shared services are extracted only when behavior is preserved by regression tests.

### S1 — Studio shell and workspace choice

Status: **in progress**

Acceptance:

- A professional Studio dashboard is available as a separate Streamlit page.
- Users can move explicitly between Studio and Classic.
- The mature Process Flowsheet Studio is the first enabled Studio engineering workflow.
- Planned workflows remain visible with honest availability state.

### S2 — Shared case context and lifecycle

Status: **in stack on top of S0/S1**.

Create/open/save/save-as/reset, metadata, units, thermodynamic package, provenance
and solved/dirty/error state shared across Studio pages without corrupting Classic cases.

Stack evidence:

- a UI-independent `studio.case_context` owns only Studio session lifecycle metadata;
- schema-v1–v4 Process Flowsheet Studio JSON remains the authoritative portable
  case contract and is not wrapped or silently rewritten;
- New, Open, Download, Save As, Reset and session-local Recent Cases are available
  from the Studio home and hand off to the existing detailed v4 importer;
- the active case carries stable identity, units, thermodynamic summary,
  provenance, portable specification and explicit draft/dirty/solved/warning/
  failed/timed-out/invalid state across Studio pages;
- solved native model availability and signature are recorded as runtime evidence,
  while Java/Python model objects remain in the established session adapter;
- lifecycle keys are Studio-owned and regression-tested not to alter unrelated
  Classic session data.

### S3 — Integrate mature Process Flowsheet Studio

Status: **initial handoff in stack; existing simulation functionality already substantial**.

Repackage the existing editor/builder/diagnostics/workbook functionality behind the
Studio shell while preserving the existing direct Classic-compatible page.

The active Studio case now hands its established solved `process_model` directly
to Process Chat. The existing flowsheet page and Process Chat page remain usable
through their original Classic routes.

### S4 — Professional results, design and engineering studies

Status: planned integration.

Expose solved/warning/failed state, conservation, assumptions, provenance,
case comparisons, design limits, sensitivity, adjust/specification and optimization
through the active Studio case.

### S5 — Engineering Drawings

Status: dependent on merged core capabilities.

Consume the canonical PFD/P&ID/DEXPI model from `equinor/neqsim`, including stable
semantic IDs, drawing sets/sheets, validation diagnostics and export artifacts.
The web layer owns interactive engineering workflow, not standards semantics.

### S6 — Process Chat engineering copilot

Status: **initial case-aware integration in stack**.

Make Process Chat case-aware and tool-oriented while keeping deterministic NeqSim
calculations as the source of truth.

The first integration exposes Process Chat as an available Studio workflow,
identifies the active case and its thermodynamic/lifecycle state, and reuses the
existing solved-model session handoff. If Process Chat replaces or resets the live
model, the shared context is marked dirty and the portable flowsheet specification
remains visible as the last reproducible input rather than being silently promoted.
Process Chat also receives a bounded, whitelisted projection of the active case ID,
lifecycle, units, thermodynamic package, runtime-model evidence and provenance. The
projection excludes the portable case body and arbitrary session state, treats every
user-authored value as untrusted data, and tells the assistant never to present a
draft/dirty/failed case as solved. Numeric results remain sourced from the live model
or an executed deterministic NeqSim calculation.

### S7 — Dynamics and controls

Status: dependent on validated NeqSim core handoff/dynamics.

### S8 — Large ProcessModel / multi-area workspace

Status: planned.

### S9 — Production hardening

Status: ongoing across all milestones.

Browser/end-to-end tests, accessibility, performance, safe execution isolation,
migration/interoperability evidence and deployment validation are release gates.

## Current tranche

The first Studio campaign tranche establishes S0/S1 without rewriting existing
simulation functionality:

1. preserve the Classic homepage and add a small explicit workspace choice;
2. add the separate professional `NeqSim Studio (Beta)` dashboard;
3. route the first available Studio workflow to the existing validated Process
   Flowsheet Studio;
4. add regression coverage for the Classic/Studio separation and declarative
   Studio destination contract.

Only merged pull requests count as completed roadmap evidence.

## Active stacked work

The S0/S1 Studio shell remains the root PR targeting `main`. The S2 child PR
targets that exact Studio branch and is not independently mergeable to `main`.
S2 advances the complete case-workspace handoff while reusing the existing v4
schema validation, migration, graph draft, native solve and Process Chat state.
Its work is recorded as **in stack**, not merged completion.

The next stacked child adds the case-aware Process Chat handoff on top of S2. It
does not expand chat permissions or duplicate simulation tools; it makes the
already-validated model transfer and lifecycle boundary explicit.
