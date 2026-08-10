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

Status: **professional results/design merged; studies and engineering evidence in recovery PR**.

The current tranche adds a shared, UI-independent exact-result handoff and a
professional results workspace over the existing solved model. It exposes:

- explicit solved/warning/attention state without presenting stale or dirty results;
- KPIs, units, thermodynamics, assumptions and solver provenance;
- streams, equipment, design capacities, margins, utilization and limit status;
- system/component/unit conservation and recycle/adjuster convergence evidence;
- display-safe session case comparison while retaining sensitivity, adjust and
  bounded optimization execution in the inherited flowsheet tools.

Professional results and design review are merged on `main` through PR #96.

The dependency-ordered child tranche adds exact-model engineering-study evidence:
completed Process Chat sensitivity sweeps and bounded optimization results are
projected into the active Studio case only when the chat session references the
same native model and solved signature. Failed points, units, method, convergence,
bottlenecks and provenance remain visible. Study execution stays in the existing
NeqSim-backed tools. Retained study evidence survives ordinary follow-up turns,
while Process Chat resets its chat-owned session and attachments if a newly solved
flowsheet replaces the live runtime model.

The third stack tranche extends the same presentation adapter to completed
scenario comparison, emissions and energy-audit attachments. Successful and
failed scenarios, KPI units, constraint/patch evidence, source-level emissions,
utility consumers, benchmark status and screening recommendations remain visible
for the exact active native model. Studio does not rerun or duplicate these
calculations, and estimation/benchmark methods remain explicitly labelled.

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

PR #96 merged the S4 professional active-case results root onto `main`. PRs #98
and #99 were then merged into their already-merged parent branches, so GitHub
closed them without landing their study/evidence trees on `main`. The current
recovery tranche reapplies their previously reviewed exact-model sensitivity,
optimization, scenario, emissions and energy-audit presentation as one clean
commit based on the exact current `main` head.

Acceptance requires byte-equivalent application/test content to validated PR #99
head (apart from this truthful roadmap update), zero Classic route or calculation
changes, fail-closed native-model/signature checks, and a fresh complete hosted
Classic + Studio + native-conservation gate.

## Active stacked work

PR #96 is merged on `main`. Closed PRs #98 and #99 preserve the original review,
exact-head validation and dependency history, but their child-branch merges did
not update `main`. One clean replacement root PR now targets current `main` with
the combined #98/#99 tree; it is recorded as **in stack**, not merged completion.

After recovery lands, the next dependency-ready tranche is a core-API-aligned
Engineering Drawings adapter and drawing register using the merged #1332/#2899
contracts. The web layer must not duplicate canonical diagram semantics.
