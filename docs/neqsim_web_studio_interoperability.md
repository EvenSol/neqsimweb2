# NeqSim Studio case interoperability contract

This document defines the supported portable case boundary for NeqSim Studio.
It records application behavior; it does not introduce a second simulation
format or replace NeqSim's native model serialization.

## Formats and ownership

| Format | Owner and purpose | Compatibility promise |
| --- | --- | --- |
| Process Flowsheet Studio JSON | NeqSim Web input case, graph, units, connections, explicit engineering units and editable assumptions | Schema versions 1–4 are accepted and exported as canonical schema version 4 |
| Native `.neqsim` archive | NeqSim Java/Python solved `ProcessSystem` or `ProcessModel` state | Read and written through the deployed NeqSim runtime; not interchangeable with portable JSON |
| Studio session context | NeqSim Web active-case identity, lifecycle, provenance and live-model binding | Session-local only; it is not embedded into or required by a Classic-compatible portable case |

The portable JSON remains authoritative for editable Studio case inputs. Native
NeqSim remains authoritative for calculations and solved process objects.

## Portable schema matrix

| Input | Import result | Export result | Case identity | Failure behavior |
| --- | --- | --- | --- | --- |
| Schema v1 | Migrated through the existing flowsheet importer | Canonical schema v4 | External open receives a new Studio identity | Not applicable |
| Schema v2 | Migrated through the existing flowsheet importer | Canonical schema v4 | External open receives a new Studio identity | Not applicable |
| Schema v3 | Migrated through the existing flowsheet importer | Canonical schema v4 | External open receives a new Studio identity | Not applicable |
| Schema v4 | Opened without schema migration | Canonical schema v4 | External open receives a new Studio identity | Not applicable |
| Future schema v5 or later | Rejected | No replacement export is published | Existing active case and Recent Cases remain unchanged | Fail closed with the supported-version diagnostic |
| Malformed JSON | Rejected before handoff | No replacement export is published | Existing Studio and Classic state remain unchanged | Fail closed with a JSON diagnostic |
| Non-UTF-8 bytes | Rejected before handoff | No replacement export is published | Existing Studio and Classic state remain unchanged | Fail closed with an encoding diagnostic |
| File larger than 1,000,000 bytes | Rejected before decode | No replacement export is published | Existing Studio and Classic state remain unchanged | Fail closed with the size-limit diagnostic |

UTF-8 with an optional byte-order mark is supported. Unicode case names are
preserved. JSON serialization rejects non-finite numeric values.

## Identity and restart rules

- Opening an external portable case creates a new Studio case identity even when
  the case body matches an existing case.
- Continuing the active case or opening a session-local Recent Case preserves its
  existing Studio identity.
- Saving As creates a distinct case identity and portable case fingerprint.
- Canonical schema-v4 downloads can be opened in a fresh browser session and
  exported again without changing the portable case body.
- Active runtime objects and solved signatures are deliberately not placed in the
  portable JSON. A reopened case must be solved again before results are treated
  as current engineering evidence.
- Rejected imports consume no pending handoff and must allow a supported retry.

## Classic compatibility

Studio import, migration, download, Save As, Recent Cases and reset operations own
only Studio state. They do not mutate Classic page state or require Classic users
to migrate saved work. The existing Process Flowsheet Studio route remains
available directly.

## Browser acceptance evidence

The retained Chromium interoperability matrix exercises:

1. schema-v1 through schema-v4 upload through the real Studio file control;
2. migration to canonical schema v4 and portable download;
3. Classic/Studio switching followed by continuation of the same active case;
4. reopening the canonical export in a fresh browser context;
5. malformed JSON, non-UTF-8, oversized and future-schema rejection;
6. a supported schema-v1 retry after every rejected input;
7. repeated application health checks and absence of browser page errors.

The browser test complements the UI-independent lifecycle and AppTest migration
regressions. It does not claim compatibility with an untested future schema or
with arbitrary native `.neqsim` archives from different NeqSim versions.

## Isolated execution boundary

Current native construction, solving, result extraction and serialization are
fail-loud and bounded by explicit caller deadlines. Some waits still use worker
threads in the Streamlit process. A timed-out model is discarded, but Python
cannot guarantee termination of every native call in that same process.

Production support for untrusted or materially larger cases therefore remains
blocked on a process-isolated worker contract that can terminate the complete
JVM worker, return only validated portable evidence and native artifacts, and
rehydrate a solved model without silently repeating or trusting an incomplete
calculation. This browser-matrix tranche does not weaken that boundary or present
thread timeouts as process isolation.

## Scope and downstream reuse

The contract preserves stable unit, stream and connection identifiers already
present in the case schema, explicit units, portable provenance and deterministic
round trips. These properties are reusable by later external acceptance
campaigns, but no Huldra dataset, mapping or model is included here.
