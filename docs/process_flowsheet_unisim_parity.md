# Process Flowsheet Studio: UniSim-parity assessment

This living assessment records evidence at the 90-commit boundary. “Partial”
means the workflow is useful for engineering studies but is not intended to
claim full UniSim capability or design certification.

| Area | Current evidence | Status | Next production gate |
| --- | --- | --- | --- |
| Thermodynamics and characterization | Shared EOS/mixing-rule/component package with independent inlet conditions and strict component compatibility | Partial | Binary-interaction and pseudo-component editing with reference validation |
| Unit operations | Native streams, separators, compressors, pumps, heaters/coolers, two-sided exchangers, valves, mixers, splitters, expanders and pipelines | Partial | Validated columns, absorbers, utilities and packaged templates |
| Graph execution | Explicit named material/energy connections, multiple feeds/products, phase outlets and subflowsheet boundaries | Strong steady-state basis | Broader boundary-port execution and scale benchmarks |
| Recycles and convergence | Native recycle/adjuster diagnostics, closure evidence and fail-loud worker status | Partial | Isolated worker process and richer tear-stream controls |
| Equipment design | Separator sizing, compressor limits, pump, exchanger, valve and pipeline design screens/datasheets | Partial | Closed multi-equipment design reruns and standards-based checks |
| Dynamics and controls | Dynamic handoff remains downstream of steady-state Studio | Gap | Validated steady-state-to-dynamic initialization and controller mapping |
| Optimization | Sensitivity, adjust/specification and bounded study tools exist | Partial | Operability-aware nonlinear optimization with discrete choices |
| Reporting | Workbook streams, equipment, constraints, convergence, balances, design and subflowsheet interfaces | Strong | Report templates, traceable assumptions and comparison packs |
| Interoperability | Process Chat solved-model handoff, native `.neqsim` persistence and schema migration | Strong | Stable external case/API contract and round-trip compatibility matrix |
| Usability | Searchable palette, graphical editing, history, explicit units, unsolved-draft persistence, semantic/mobile shell and in-stack real-browser workspace-switching evidence | Partial | Complete browser solve/export/Process Chat pilots and formal WCAG audit |
| Performance | Hosted/local gates, native nearby points, 500-unit/large-result projection budgets and Chromium paint/network/heap baselines | Partial | Multi-session concurrency, failure isolation and evidence-driven optimization |
| Validation evidence | Native baseline/+5% conservation, convergence, serialization, workbook and Streamlit health gates | Strong for covered units | Independent reference cases and expanded pilot acceptance |

## Production-ready steady-state v1 priorities

1. Keep native execution fail-loud and time-bounded; move long-running solves
   into fully isolated workers before supporting untrusted large cases.
2. Add representative small, medium and large graph performance benchmarks
   before introducing result caching.
3. Extend browser-level case open, edit, solve, export and Process Chat handoff
   tests while retaining native mass, component and energy closure gates.
4. Publish a stable migration and interoperability matrix for schema and
   `.neqsim` metadata revisions.
