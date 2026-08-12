# NeqSim Web Studio Engineering Drawings integration contract

## Purpose

This document defines the boundary for Studio Engineering Drawings without
inventing diagram semantics in `EvenSol/neqsimweb2`. It records verified merged
core evidence, the missing activation dependencies and the web-layer acceptance
contract.

`equinor/neqsim` owns canonical engineering topology, semantics, stable IDs,
content profiles, validation/loss evidence and rendering/export behavior.
`neqsimweb2` may present those outputs, link them to the active Studio case and
manage user interaction. It must not reconstruct PFD/P&ID/DEXPI meaning from UI
state or a simulation-only graph.

## Verified core baseline

The merged baseline is pinned to NeqSim PR #2960 at
`e8991c41163018299c0eacf8fef96e378a8fca72` and PR #2961 at
`7402265158f18a40f6c71bf94fb033557ed263f7`:

- canonical `ProcessSystem` and multi-area `ProcessModel` topology projection with
  stable plant, area, equipment, port and connection identity;
- explicit material, energy and signal topology in the canonical graph where
  supported;
- structured diagnostics for unsupported or ambiguous projection behavior;
- canonical-topology-backed assessed DEXPI 2.0 Process material export through
  merged PR #2932;
- opt-in successful-run operating-case snapshots through merged PR #2934, with
  stable case/object identity, temperature in K, absolute pressure in bara, mass
  flow in kg/s, simulation provenance and review-required status;
- deterministic simple, branched and multi-area drawing reference cases through
  merged PR #2960;
- immutable `EngineeringDiagramDocumentSet` drawing-register and sheet semantics
  through merged PR #2961, including stable drawing/sheet/object IDs,
  revision/status/issue-purpose metadata, reciprocal off-page connectors,
  structured diagnostics and deterministic JSON/fingerprints;
- a core `ProcessDiagramDocumentSetAdapter` for both `ProcessSystem` and
  `ProcessModel`, with multi-area sheet ownership and preserved material, energy
  and signal connector semantics;
- legacy DOT/Graphviz and compatibility DEXPI/Proteus paths preserved separately.

This baseline now supplies the controlled semantic document set that Studio must
consume. It deliberately does not supply layout/routing, symbols or title blocks,
manual layout overrides, qualified SVG/PDF artifacts, or a native DEXPI
document/graphics projection.

## Dependencies that must remain in core

Studio activation still requires merged, documented and runtime-available APIs
for the following capabilities:

1. A deployed NeqSim runtime exposing the merged PR #2961 document-set API through
   a stable Java/Python integration path.
2. Exact solved-case binding and governed operating/design metadata sufficient to
   prove that a drawing belongs to the active Studio solution, with explicit
   units and provenance.
3. Core-owned layout, routing, symbol/title-block semantics and any supported
   manual-override persistence.
4. Core-produced SVG/PDF/DEXPI artifacts with media type, content profile,
   checksum/provenance and structured validation/loss diagnostics.
5. Multi-area artifact behavior that preserves hierarchy, reciprocal off-page
   references and material, energy and signal connections.

Selected operating values with explicit units, stable case/object identity and
provenance are merged in core PR #2934. The controlled document/sheet model is
merged in PR #2961. Core PR #2966 is an active, unmerged metadata increment and is
not accepted as Studio capability. Artifact generation and qualification remain
outside the merged document-set scope. A merged core commit is also insufficient
when the deployed web runtime still uses a NeqSim release that lacks the API.

## Studio adapter responsibilities

Once the dependencies are merged and available in the runtime, the Studio adapter
may:

- load one immutable drawing-set result associated with the exact active case and
  solved signature;
- display the drawing register, sheets, revision/status, artifacts and core
  diagnostics without changing their engineering meaning;
- navigate sheets and select stable equipment, stream and connector IDs;
- link a selected drawing object to the matching flowsheet object and professional
  results evidence;
- download core-produced SVG, PDF and DEXPI artifacts unchanged;
- retain explicit provenance, units, unsupported-scope diagnostics and failure
  state;
- keep legacy Classic Graphviz/diagram routes unchanged.

The adapter must fail closed when case identity, solved signature, stable object
identity, artifact integrity or required diagnostics are missing. It must not
infer missing pipes, valves, nozzles, instruments, safeguards or cross-sheet
connections.

## Presentation and claim boundary

- A simulation-driven PFD may be presented when the core output identifies its
  content profile and validation status.
- P&ID output remains labelled proposal or preliminary until accountable piping,
  valve, nozzle, instrument and safeguard data are present and reviewed.
- No Studio page may claim ISO 10628, ISO 14617, DEXPI or project-standard
  conformance merely because an artifact renders. Qualification remains tied to
  the core capability report and accountable review.
- Missing or lossy mappings remain visible; blank output is a failure, not a
  successful drawing.

## Acceptance gate for the first Studio drawing tranche

The first implementation PR must demonstrate all of the following against its
immediate base and the cumulative Studio stack:

- exact-case and exact-solved-signature binding;
- deterministic drawing-register and sheet metadata mapping;
- stable ID selection between drawing, flowsheet and results views;
- structured validation/loss/error presentation;
- unchanged artifact bytes on download;
- multi-area and cross-sheet behavior covered by a merged core reference case;
- Classic diagram routes and saved cases unchanged;
- fresh Streamlit root and health probes plus focused browser interaction tests;
- the canonical NeqSim-Colab DEXPI safety-study workflow remains executable, or is
  updated separately only when a merged substantial capability lacks acceptance
  evidence.

Until this gate is implementable from merged runtime APIs, Engineering Drawings
must remain visible but disabled as core integration in progress.
