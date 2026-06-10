# Changelog

## v1.2 — Pipeline-friendly CLI, resource governance, docs (unreleased)

Hardens the standalone binary into a well-behaved pipeline trainer (GSplata,
OpenSplat-style harnesses) and adds the test/doc/benchmark scaffolding the project
lacked. No change to the training math — the kernels are byte-for-byte v1.1.3.

**Process contract (observable & controllable)**
- **Streaming progress** on stdout, line-buffered and **off-TTY** (so it shows when
  piped): `--progress-every N` (default 100) →
  `step=i/n splats=C loss=L ms/it=T`, with `--progress-format {plain,jsonl}` for a
  parser-stable variant. The reported loss is real (a cheap pipeline drain at each
  progress cadence; <1% overhead).
- **Final summary line**: `Done: <n> iters, <count> Gaussians, PSNR <x>, wrote <abs>`.
- **Signal handling**: `SIGINT`/`SIGTERM` finish the in-flight iteration, write
  `<output>_interrupted.ply`, and exit 130/143. `SIGSTOP`/`SIGCONT` via the OS.
- **Deterministic exit codes**: 0 ok · 3 dataset-load · 5 write · 130/143 signals.
- `--quiet` / `--verbose`.

**Resource governance (runs well on a base M4 / 16 GB)**
- **`--max-splats N`** hard cap — above the cap the densify pass prunes only
  (growth disabled by raising the effective gradient threshold), bounding peak
  unified memory with no kernel change. Validated: cap 50k converges to 46,799
  vs 100,883 uncapped (garden, `-n 2000 -d 4`).
- **`--memory-budget GB`** advisory → derives `--max-splats`.
- **Default `--bg-color` is now black** (pipeline-safe; matches viewers). The old
  magenta debug background is available via **`--debug-bg`**. *Measured: black vs
  magenta are within noise on garden 7K (24.98 vs 24.95 PSNR), so the default change
  has no meaningful quality cost.*

**UX & packaging**
- **`--preset {draft,balanced,production}`** (7000/-d2/sh2 · 30000 · 100000, with
  matching `--max-splats`); explicit flags override the preset. Grouped `--help`.
- **`man` page** `docs/msplat.1`, installed to `share/man/man1`.
- **Shell completions** (bash + zsh) in `scripts/completions/`, installed by
  `cmake --install` to `share/bash-completion/completions` and
  `share/zsh/site-functions`; complete enum values for `--preset`/`--progress-format`
  and paths for file/dir flags.

**Tests, build, docs**
- **C++ unit tests** (doctest, header-only, no GPU): progress-line format, downscale
  schedule, SH basis counts, budget heuristic, and a PLY-header round-trip that
  guards the `--resume` step-comment prefix. `ctest` target, default ON.
- Testability refactor: pure shared logic moved to `core/include/cli_support.hpp`
  and `core/include/ply_header.hpp` (single tested source for CLI + model + loaders).
- **`scripts/build.sh`** — robust Release/Debug build with a Metal-toolchain
  preflight, runs the unit tests, prints artifacts.
- **`docs/`** — user getting-started, dev building/testing + internals +
  profile-driven optimization roadmap (with cited 3DGS research), and a dated
  benchmark ledger. **`gsplata_integration.md`** documents the now-satisfied trainer
  contract.

**Fixes**
- **`--resume` lost its step**: the writer emitted `comment msplat v<step>` but the
  reader parsed a stale prefix, so resuming restarted the LR/densification schedule
  from 1. Reader and writer now share the prefix (unit-test guarded).
- Tightened the PLY magic check to require a leading `ply` token.
- **PLY loader validates the header vertex count against file size** before
  allocating, so a corrupt/truncated/hostile header is rejected with a clear error
  instead of attempting a huge allocation.

**Robustness (training stability)**
- **Coarse-render warning**: training small-image datasets at an aggressive
  `--downscale-factor` (e.g. Tanks&Temples 979px at `-d4` → 244px render) destabilizes
  the optimizer and collapses to an empty render. The CLI now warns when the render
  long-side is < 400px and suggests a smaller `-d`. (`camera_math.hpp`, unit-tested.)
- **NaN/Inf gradient guard** in the Adam updates (`adam_update_element` +
  `fused_adam_kernel`): a non-finite gradient is zeroed so it can't poison a
  parameter and propagate. Safety net (no effect on healthy scenes).
- Verified + unit-tested the COLMAP intrinsics rescale (declared SfM resolution vs
  actual downscaled images) — it was already correct.
- Extended the C++ unit suite to 11 cases (intrinsics rescale, coarse-render guard).

**Validated on Mac16,1 (MacBook Pro M4, 16 GB)** — the constrained target machine:
garden 7K (`-d 1`, default progressive schedule) trains end-to-end in **174 s at
4.86 GB resident** (black bg 24.98 / magenta 24.95 PSNR — refactor is non-regressing;
the two backgrounds are within noise). Matching the README command exactly
(`--num-downscales 0`, full-res throughout) reproduces **25.67 / 0.7835** vs the
v1.1.3 README's 25.68 / 0.783 — i.e. **no quality regression** from the v1.2 work.

## v1.1.3 — Fused kernels + pre-allocated tile bins

- **Fused SH backward into Adam optimizer** — spherical harmonics gradients are now
  computed in registers and fed directly into Adam updates, eliminating a ~600 MB/iter
  device memory round-trip (at 1.5M gaussians).
- **Fused SSIM vertical-forward + horizontal-backward** — replaces two separate passes
  with a single kernel that recomputes V-conv from the H-buffer, saving ~130 MB/iter
  of intermediate buffer traffic.
- **Pre-allocated per-tile bins** — replaces the count→prefix-sum→scatter intersection
  pipeline with direct scatter to fixed-size per-tile bins. Eliminates 3 kernel
  dispatches and 3 memory barriers per iteration. `prefix_sort_pack` stage reduced
  from 19% to 10% of GPU time.
- **14–48% faster training** across mipnerf360 scenes. Improvement scales with gaussian
  count: garden 30K (3.5M gaussians) sees the largest speedup at 48%.
- **Per-stage GPU profiling** — `PROFILE_STAGES=1` enables Metal timestamp counter
  sampling per pipeline stage. Uses separate compute encoders on the same command buffer
  with `MTLComputePassDescriptor` for zero-overhead timestamp capture.
- **GPU timing instrumentation** — `PROFILE_GPU=1` adds completion handler timing to
  command buffers, reporting per-CB GPU execution time without affecting the
  `commitAndContinue` pipeline.

## v1.1.2

- Added `py.typed` marker (PEP 561) — type checkers now discover stubs automatically
- `TrainingConfig(bg_color=...)` now raises `ValueError` on wrong-size lists instead
  of silently falling back to the default

## v1.1.1

- Fixed `new[]`/`free()` mismatch in C API pixel buffer allocation — undefined
  behavior when Swift or other C callers freed render output with `free()`.
  Allocation now uses `malloc` consistently.
- Updated type stubs (`_core.pyi`) with `camera_pose` and `render_from_pose`
  methods added in v1.1.

## v1.1 — Arbitrary viewpoint rendering

- **`renderFromPose` API** — render from any camera-to-world matrix, not just dataset cameras.
  Uses intrinsics from a reference camera. Available across all surfaces:
  - C++: `trainer.renderFromPose(camToWorld, refCameraIndex)`
  - C API: `msplat_trainer_render_pose()`
  - Python: `trainer.render_from_pose(cam_to_world, ref_cam_idx=0)`
  - Swift: `trainer.renderFromPose(camToWorld:refCameraIndex:)`
- **`renderFromPoseToBuffer`** — zero-copy variant that writes directly into a
  caller-provided RGBA uint8 buffer. Eliminates intermediate float allocation for
  real-time display loops (400 FPS at full resolution on M4 Max).
  - C++: `trainer.renderFromPoseToBuffer(camToWorld, ref, outRGBA, &w, &h)`
  - C API: `msplat_trainer_render_pose_to_buffer()`
- **`cameraPose` accessor** — retrieve camera-to-world matrices from loaded datasets.
  - C++: `dataset.cameraPose(index, outMatrix)`
  - C API: `msplat_dataset_camera_pose()`
  - Python: `dataset.camera_pose(index)` → numpy `(4, 4)` float32
  - Swift: `dataset.cameraPose(at: index)` → `[Float]`
- **Demo app** (`demo/`) — macOS SwiftUI app for screen-recording hero videos.
  Live training with progress bar, then smooth circular camera orbit with FPS counter.

## v1.0 — Public release

Stable API across Python, Swift, and C++ surfaces.

## v0.6 — Bug fixes and API improvements

- Fixed SSIM Gaussian kernel — ported formula `floor((i - windowSize) / 2.0)`
  produced pairwise-duplicated values instead of a symmetric bell curve.
  Corrected to `i - windowSize / 2` in both Metal shader and CPU eval path.
- Fixed ASCII PLY reader — x coordinate used byte offset instead of token index,
  silently reading wrong values when x isn't the first property.
- Background color now configurable across all APIs (Python, Swift, C++, CLI).
  Default magenta `[0.613, 0.010, 0.398]` documented as intentional
  (high contrast for debugging under-reconstructed regions).
  - Python: `TrainingConfig(bg_color=[r, g, b])`
  - Swift: `config.bgColor = (r, g, b)`
  - CLI: `--bg-color R G B`
- `cleanup()` now safe to call multiple times (Python guard prevents double-free
  when manual call + atexit handler both fire)
- Added type stubs (`_core.pyi`) — IDEs now have autocompletion and type checking
  for the compiled extension module
- Documented `MTensor.view()` use-after-free risk (non-owning alias)

## v0.5 — Open-source cleanup

- Removed datasets from git (1+ GB of LFS-tracked files)
  - CI/release workflows now download garden dataset from Google Storage with caching
- Code quality fixes
  - `exit(1)` on image load failure → `throw std::runtime_error` (safe for library consumers)
  - Deduplicated `getCachedMTensorImage` (3 copies) into `Camera::getGPUImage()` method
  - Removed debug `printf` on metallib load, commented-out `printf` in Metal shader
  - Deleted dead `msplat_model.hpp` alias header
  - Error messages to `stderr` instead of `stdout`
- Python API improvements
  - Removed always-zero `loss`/`psnr` fields from `TrainingStats`
  - Added docstrings to all nanobind bindings (TrainingConfig, TrainingStats, Dataset, GaussianTrainer)
  - Fixed `requires-python` from `>=3.10` to `>=3.12` (only supported versions)
  - Fixed SPDX license / classifier conflict in `pyproject.toml`
- Swift package: added render and export PLY tests (3 → 5 tests)
- Apache 2.0 license
- Full PyPI metadata (author, classifiers, keywords, URLs)

## v0.4.1

- Swift XCFramework distribution: `scripts/build-xcframework.sh` builds a self-contained XCFramework
  - `msplat_set_metallib_path()` C API for explicit Metal library path configuration
  - Swift wrapper auto-configures metallib via `Bundle.module`
  - Replaced CMsplat bridge target with `.binaryTarget` pointing at XCFramework
- GitHub Actions CI/CD
  - `ci.yml`: build + test C++ CLI, Python wheels (3.12/3.13), Swift package on every push
  - `release.yml`: GitHub Releases + PyPI publishing on tagged commits
  - Version sync check (VERSION, pyproject.toml, `__init__.py`) gates all jobs
- Fixed OpenGL Y/Z flip in COLMAP pose conversion (negate columns, not rows)
- Removed `constants.hpp` — `APP_VERSION` from CMake, `PI` → `M_PI`

## v0.4 — Drop OpenCV dependency

- Replaced OpenCV with lightweight built-in implementations
  - `Image` struct (float32 RGB) replaces `cv::Mat` throughout
  - Area-based image resize (box filter) replaces `cv::resize(INTER_AREA)`
  - Brown-Conrady undistortion with alpha=0 crop replaces `cv::undistort`
  - CoreGraphics PNG writing replaces `cv::imwrite`
  - Dropped dead Linux/OpenCV fallback code (Metal is macOS-only)
- No external dependencies beyond system frameworks (Metal, CoreGraphics, ImageIO)
- Removed `brew install opencv` requirement

## v0.3 — Checkpoint system, clean-room loaders, CLI11

- Checkpoint save/resume (`trainer.save_checkpoint()` / `trainer.load_checkpoint()`)
  - Binary `.msplat` format: gaussian params + full Adam optimizer state
  - Bound in Python, Swift, and C API
  - 2 new tests (save/load + resume round-trip)
- Rewrote all dataset loaders from scratch
  - COLMAP binary format (cameras.bin, images.bin, points3D.bin)
  - Nerfstudio transforms.json
  - Polycam (keyframes/ and cameras.json layouts)
  - Dropped OpenSfM + OpenMVG (low adoption, trivially convertible to COLMAP)
  - PLY point cloud reader + COLMAP binary point reader
  - CoreGraphics image loading on macOS
- Moved Gaussian PLY/splat I/O out of model.cpp into `loaders/save_gaussians.cpp`
- Switched CLI from cxxopts to CLI11 (validation, subcommand-ready)
- Rewrote `utils.hpp` → `random_iter.hpp` (dropped `parallel_for`)
- Made `kdtree_tensor` header-only
- Removed `tensor_math.{cpp,hpp}` (unused)
- Loader code reorganized into `core/src/loaders/` subdirectory

## v0.2 — Swift Package + general cleanup

- Swift Package with C API bridge (3 tests: config, dataset loading, 10-step training)
- C API header (`msplat_c_api.h`) for Swift interop via opaque handles
- `msplat_api.{hpp,mm}` compiled into `libmsplat_core.a` (not SPM)

## v0.1 — Initial release

Standalone 3D Gaussian Splatting engine for Apple Silicon with 44 fused Metal
compute kernels and Python bindings via nanobind.

- C++ core with Metal backend (44 fused compute kernels)
- CMake build system: `libmsplat_core.a` static library + `msplat` CLI
- Python package (`pip install msplat`) via scikit-build-core + nanobind
- Full training pipeline: `GaussianTrainer.train()` with progress callbacks
- Multi-format dataset loading: COLMAP, Nerfstudio, OpenSfM, OpenMVG
- Evaluation on held-out test views (PSNR, SSIM, L1)
- PLY and .splat export
- Rendering API: `trainer.render(cam_idx)`
- Python CLI: `msplat-train path/to/dataset -n 7000 --eval`

### Numbers

Garden (mipnerf360), 7K steps, 24 test views:
- PSNR: 25.75 dB
- SSIM: 0.786
- 1.5M gaussians
- ~3 ms/iter at 4x downscale, ~17 ms/iter full res (M4 Max)
