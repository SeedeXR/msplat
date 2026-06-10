# Internals

A condensed developer map. The exhaustive, line-cited version lives in
[`memory/architecture.md`](../../memory/architecture.md).

## Layout

```
core/metal/msplat_metal.metal   44 fused Metal kernels (compiled metal -O3 → default.metallib)
core/metal/msplat_metal.mm      Metal host: buffer cache, dispatch, command-buffer lifecycle
core/src/model.cpp              training orchestration, schedulers, densification policy, save/load
core/src/loaders/               COLMAP / Nerfstudio / Polycam / PLY / image I/O / save
core/src/ssim.cpp               CPU SSIM (eval metrics only)
core/include/                   MTensor, Model, APIs, and the pure helpers (cli_support, ply_header)
cli/msplat.cpp                  the standalone binary (CLI11) — a thin shell over Model
python/ swift/ demo/            nanobind module · Swift package (C API) · SwiftUI demo
```

Three thin "doors" (CLI, Python/nanobind, C-API→Swift) over one engine. Doors
marshal config in and tensors/files out; capability lives in the core. The shared
pure helpers (`cli_support.hpp`, `ply_header.hpp`) hold logic that must be
identical across doors and is unit-tested without a GPU.

## The per-iteration GPU pipeline

One training step encodes all work into a single `MPSCommandBuffer` and ends with
`commitAndContinue()` (non-blocking, pipelines across iterations):

```
blit-zero loss_sum / counters       (GPU-side, no CPU memset)
project_and_sh_forward              fused 3D→2D projection + SH color
scatter_to_prealloc_bins            (tile_id<<16 | depth16) keys into per-tile bins
prefix_sum (+2 helpers)             hierarchical scan of tile counts
bitonic_sort_per_tile               256 thr/tile, sorts ≤2048/tile, packs xy/conic/rgb inline
nd_rasterize_forward                8×8 threads/tile alpha compositing, early-exit
  (chunked + merge path for very deep tiles)
ssim_h_fwd → ssim_v_fwd             separable 11-tap SSIM + L1 → atomic loss_sum
rasterize_backward                  per-pixel backward, SIMD reduce → atomics
project_and_sh_backward_adam        FUSED projection-bwd + SH-VJP + Adam update in registers
fused_adam ×4 groups                means / scales / quats / opacity
accumulate_grad_stats               gradient norms + visibility for densification
```

Design principles (see [`memory/philosophy.md`](../../memory/philosophy.md)):
the GPU owns the loop; fuse to kill device-memory round-trips; **zero per-iteration
allocations** (a buffer cache resizes only at densification); minimize bytes moved.

### Loss readback is sync-free
`msplat_train_step` reads `loss_sum` *without* a GPU sync, so the value is lagged
by ~1 iteration but free. The CLI only forces a real read at progress cadence via
`msplat_read_loss()` after a `msplat_gpu_sync()` (the buffer is blit-zeroed each
step, so a sync-free read races the zero-fill and returns 0 — this is why earlier
versions removed the "always-zero loss" field). Cost: one pipeline drain per
`--progress-every` steps (<1% at the default 100).

## Densification (every `refine-every`=100 after `warmup-length`=500)

GPU-resident classify → split/clone → cull → compact. Policy lives in
`Model::afterTrain` (model.cpp): grad ≥ `densify-grad-thresh` splits/clones (by
`densify-size-thresh`), opacity < 0.1 culls, opacity resets every
`reset-alpha-every` refines. Growth stops at `maxSteps/2`.

**`--max-splats` cap:** at or above the cap, `afterTrain` raises the effective
gradient threshold to `FLT_MAX`, so the densify pass *prunes only* — culling still
reclaims dead gaussians while growth halts. No kernel change; bounds peak memory.

> Memory note: `ensureCapacity` allocates new backing buffers before freeing the
> old on each grow, so transient allocation churns during densification (inflates
> `/usr/bin/time` "footprint"; resident peak stays bounded). Pre-sizing/reuse is a
> tracked optimization (roadmap M3.6).

## Memory model (unified memory)

Every `MTensor` is an `MTLBuffer` with `MTLResourceStorageModeShared` — CPU and GPU
share pages; "upload/download" is pointer access. Dominant residents: gaussian
params + 2× Adam moments (~700 B/splat ⇒ 3.5 M ≈ 2.5 GB), per-tile prealloc bins,
and the decoded image set (currently all images decoded to float32 up-front).

## Conventions (state these wherever data crosses a boundary)

- Scales are **log-space**; opacity is **pre-sigmoid**; quaternions are **[w x y z]**.
- Camera convention is OpenGL (Y-up, Z-back); COLMAP/Polycam poses are flipped on load.
- PLY: binary-little-endian, `f_rest` transposed to channel-major `[N,3,bases]`;
  the resume step rides in a `comment msplat v<step>` header line.
- float32 throughout the kernels; tile = 16×16 px, raster threadgroup 8×8,
  sort 256 threads/tile, 2048-bin per-tile cap.

## Frozen contracts (additive evolution only)

The PLY property set, the OpenSplat-compatible CLI shape, and the stdout lines a
pipeline parses (`step=…`, jsonl, `PSNR=…`, `Gaussians: N`, `Eval mode: …`,
`Done: …`) are **contracts**. Add new flags/lines; don't change existing ones.
See [`gsplata_integration.md`](../../gsplata_integration.md).
