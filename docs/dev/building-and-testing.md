# Building & testing

## Toolchain
- macOS 14+, Apple Silicon (arm64).
- Full **Xcode** + the **Metal toolchain**: `xcodebuild -downloadComponent MetalToolchain`
  (one-time, ~690 MB; `xcrun -sdk macosx metal --version` must succeed).
- CMake ≥ 3.21 (works with CMake 4.x). C++17 / Objective-C++.
- Dependencies are fetched at configure time (FetchContent): nlohmann_json 3.11.3,
  nanoflann 1.5.5, CLI11 2.4.2, doctest 2.4.11. No system packages required.

## Build

```bash
./scripts/build.sh              # Release, builds CLI + core + metallib + tests, runs ctest
./scripts/build.sh --debug      # Debug
./scripts/build.sh --clean      # wipe build/ first
./scripts/build.sh --python     # also build the nanobind extension (needs nanobind)
./scripts/build.sh --no-tests   # skip unit tests
```

Or directly:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Artifacts: `build/msplat` (CLI), `build/default.metallib` (kernels, compiled
`metal -O3`), `build/libmsplat_core.a`. The CLI loads `default.metallib` from beside
the binary; embedders can override with `msplat_set_metallib_path()`.

## Testing — three tiers

The standard: **a change is incomplete without the test tier it touches.**

### Unit (pure C++, no GPU/dataset) — fast, runs anywhere
`tests/cpp/test_units.cpp` (doctest) covers the shared logic in
`core/include/cli_support.hpp` and `core/include/ply_header.hpp`: progress-line
formatting (the exact strings the pipeline parses), the downscale schedule, SH
basis counts, the memory-budget→max-splats heuristic, and PLY-header parse +
round-trip (a regression guard for the `--resume` step-comment bug).

```bash
ctest --test-dir build --output-on-failure        # or: ./build/msplat_tests
```
Disable with `-DMSPLAT_BUILD_TESTS=OFF`.

### Integration (GPU + the in-repo garden dataset) — short trains
- Python: `pip install -e . && pytest tests/` (garden-gated: 50-step trains,
  export, checkpoint round-trip, eval sanity PSNR > 10).
- Swift: `cd swift && swift test`.
- CLI smoke (piped, no TTY — the pipeline-facing path):
  ```bash
  ./build/msplat datasets/mipnerf360/garden -n 200 -d 4 --eval \
      --progress-format jsonl -o /tmp/out.ply | cat
  ```
  Assert: jsonl lines stream *during* the run, exit 0, PLY written, `Done:` line.

### Regression (quality + speed, garden) — gate before release
Any change touching kernels/math must show garden 7K PSNR/SSIM does not regress
(> −0.05 dB needs justification) and record numbers in the
[benchmark ledger](../benchmarks/) + CHANGELOG.
```bash
./build/msplat datasets/mipnerf360/garden -n 7000 -d 1 --eval -o /tmp/g.ply
```
Keep background color fixed when comparing — it changes the metric (black vs the
old magenta default differ on garden; see the benchmark ledger).

## Profiling

Env-gated, no rebuild needed:
```bash
BENCHMARK=1                  ./build/msplat … # per-iter timing: mean/median/p5/p95, CPU vs GPU
BENCHMARK=1 PROFILE_GPU=1    ./build/msplat … # + GPU command-buffer timing
BENCHMARK=1 PROFILE_STAGES=1 ./build/msplat … # + per-kernel-stage GPU timing  ← for kernel claims
```
For deep dives use the Xcode **Metal System Trace** / GPU capture on the demo app
(`demo/`). Drive optimization with the per-stage profile — see
[optimization-roadmap.md](optimization-roadmap.md).

## CI

GitHub Actions (`.github/workflows/`) on macos-15:
1. version-sync gate (VERSION / pyproject.toml / `__init__.py` agree),
2. **C++ unit tests** (`msplat_tests` / ctest) — fast, no GPU,
3. C++ build (uploads the `msplat` binary artifact),
4. Python 3.12/3.13 wheels + pytest,
5. Swift XCFramework + tests.
Releases on `v*.*.*` tags publish the CLI binary, wheels (PyPI), and XCFramework.
