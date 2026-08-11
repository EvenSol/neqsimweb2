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

The baseline assessed against `equinor/neqsim` master commit
`c1cde6a1c47f34664c71ccab1ab2c53d85313860` is:

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
- legacy DOT/Graphviz and compatibility DEXPI/Proteus paths preserved separately.

This baseline is enough to prove a reusable semantic foundation. It is not yet a
web-consumable professional drawing package.

## Dependencies that must remain in core

Studio activation requires merged, documented and runtime-available APIs for the
following capabilities:

1. Controlled drawing-set and sheet identity, including area ownership and stable
   cross-sheet references.
2. Revision, status and approval metadata suitable for a drawing register.
3. Stable equipment, stream and connector identifiers that map back to active
   Studio case objects without name matching.
4. Explicit layout ownership and off-page connector semantics.
5. SVG/PDF/DEXPI artifacts with media type, content profile, checksum/provenance
   and structured validation/loss diagnostics.
6. Multi-area `ProcessModel` behavior that does not silently flatten hierarchy or
   lose energy/signal connections.

Selected operating values with explicit units, stable case/object identity and
provenance are now merged in core PR #2934. The opt-in adapter publishes only
successful-run values and reports unsuccessful areas instead of exposing stale
results. The assessed DEXPI 2.0 Process writer does not yet consume those value
nodes, and Studio does not consume the adapter directly. Document/sheet governance
and qualified render artifacts remain later core increments. A commit present only
on core `master` is also insufficient when the deployed web runtime still uses a
NeqSim release that lacks the API.

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
