# msplat documentation

Metal-native 3D Gaussian Splatting trainer for Apple Silicon.

## For users
- [Getting started](user/getting-started.md) — install, build, train, dataset formats, presets.
- [Datasets](datasets.md) — download training scenes from Hugging Face, per-dataset `-d` guidance, how to push new datasets.
- [CLI reference](msplat.1) — the man page (`man docs/msplat.1`); every flag, exit codes, env vars.

## For developers
- [Building & testing](dev/building-and-testing.md) — toolchain, build script, unit/integration/regression tests, profiling.
- [Internals](dev/internals.md) — architecture, the per-iteration GPU pipeline, memory model, conventions.
- [Optimization roadmap](dev/optimization-roadmap.md) — profile-driven speed/quality plan with the current per-stage GPU profile and cited research techniques.

## Integration
- [`gsplata_integration.md`](../gsplata_integration.md) — the contract for embedding msplat as a GSplata trainer.

## Benchmarks
- [benchmarks/](benchmarks/) — dated, machine-stamped measurement ledger (every perf/quality claim diffs against these).

## Project knowledge base
- [`memory/`](../memory/) — agent/contributor operating files: roadmap (`todo.md`), session log (`handover_session.md`), architecture, philosophy, and the mandatory `session_start.md` startup checklist.
