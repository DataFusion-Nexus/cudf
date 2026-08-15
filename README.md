# DataFusion Nexus cuDF fork

This repository is forked from NVIDIA [cuDF](https://github.com/rapidsai/cudf) at `v26.08.00`.

This fork keeps the upstream cuDF codebase as its base and carries local changes needed by
`datafusion-nexus` (check [diff](https://github.com/DataFusion-Nexus/cudf/compare/v26.08.00...DataFusion-Nexus:cudf:datafusion-nexus-26.08)).

## Fork Changes

The fork is intentionally limited to changes required by `datafusion-nexus`:

- GPU memory planning through retained join and regex state, plus preflight sizing for radix sort
  and fixed-width gather. Nexus-added execution APIs use explicit CUDA streams and RMM memory
  resources.
- Parquet reader fixes for predicate pruning and decimal values, plus ordered and coalesced range
  reads for remote sources.
- Shared RMM runtime identity when libcudf is built with static third-party dependencies.
