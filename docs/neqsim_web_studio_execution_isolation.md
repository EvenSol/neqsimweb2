# Studio execution-isolation contract

## Purpose

This contract defines the boundary required before NeqSim Web Studio may call a
native calculation *process isolated*. It does not replace the current bounded
execution adapter and it does not move thermodynamic or process calculations out
of NeqSim.

## Current boundary

The inherited Process Flowsheet Studio applies one explicit execution budget to
construction, native solve, result extraction, serialization and provenance.
Timed-out Java workers are interrupted where the native API permits it, the
partially mutated model is discarded, and no solved state is published.

Those controls bound the application's wait. They do not terminate the hosting
Python process or JVM and therefore are not an isolation boundary. A daemon
Python thread or interrupted Java thread may remain alive after the caller has
failed closed.

The merged NeqSim persistence APIs also do not yet provide the exact handoff that
a separate Studio worker needs:

- `ProcessSystem.loadFromNeqsim` and `ProcessModel.loadFromNeqsim` automatically
  run the deserialized model;
- the web adapter's `from_file` and `from_bytes` paths likewise rerun to
  initialize and converge the model;
- lifecycle JSON snapshots are useful for state inspection and comparison, but
  `ProcessSystemState.toProcessSystem` still documents equipment reconstruction
  as future work and returns only a new named process container.

Reusing any of those loaders in a parent process would silently execute a second
calculation or reconstruct an incomplete model. Studio must not present that as
the exact solved worker result.

## Required worker protocol

A future isolated execution service must satisfy all of the following:

1. Start a fresh Python process and JVM using a spawn-compatible entry point.
   Forking a Python process after JPype has started a JVM is outside this
   contract.
2. Accept only a validated canonical Studio case, its stable case identity,
   exact solved-signature candidate, explicit units, timeout and supported
   protocol version.
3. Build and solve through the existing `ProcessBuilder` and
   `NeqSimProcessModel` adapters. NeqSim remains the calculation source of truth.
4. Return one atomic, bounded response containing the input identity/signature,
   solver status, structured results and diagnostics, NeqSim/runtime provenance,
   and a native `.neqsim` artifact or another merged exact-state artifact.
5. Publish nothing when the worker reports failure, the protocol or signature
   differs, required evidence is missing, an artifact is invalid, or any value is
   non-finite or lacks its declared unit.
6. On timeout, terminate the complete worker process, escalate to a forced kill
   after a bounded grace period, reap it, and prove that no child remains.
7. Rehydrate the live solved model without rerunning it. This requires a merged
   NeqSim no-run restoration API or an equally authoritative exact-result
   contract with integrity and calculation-identity evidence.
8. Keep the active Studio case, Recent Cases, Process Chat handoff and unrelated
   Classic session state unchanged until the entire response has been validated.

## Acceptance evidence

Activation requires focused protocol tests plus a real native end-to-end gate:

- success, native failure, malformed response, wrong protocol, wrong case or
  solved signature, serialization failure and non-finite result rejection;
- deterministic timeout, terminate/kill escalation, worker reaping and repeated
  execution without orphan growth;
- exact result and artifact equivalence against the established in-process
  starter and conservation cases, including nearby operating points;
- safe restart and solved-model Process Chat handoff with no hidden rerun;
- two independent sessions and Classic navigation while one worker fails or
  times out;
- root HTTP 200, `/_stcore/health = ok`, a live app process and zero browser page
  errors before and after the gate.

Exploratory two-session Chromium runs showed that current in-process native
execution is not yet a deterministic release gate: one exact head completed both
solves while repeated heads lost one session beyond the same 180-second budget.
A process-local serialization experiment did not make the gate reliable and was
removed. Do not claim concurrent or process-isolated JVM execution until the
worker protocol above is available and repeatably validated. No production
execution or Classic behavior is changed by this contract.

## Stop boundary

Do not add a production subprocess launcher until the exact solved-model
rehydration dependency is merged and available in the deployed NeqSim runtime.
Do not use fork, pickle JPype proxies, trust an unverified worker artifact, rerun
on load without disclosure, or weaken the current fail-loud timeout behavior as a
workaround.
