# Optimization roadmap (profile-driven)

This is the speed/quality plan grounded in the **measured** per-stage GPU profile
on a base M4, plus implementation-ready details for current 3DGS research
techniques (with citations). Every item lists where it plugs in and the gate it
must pass. Do not land a kernel/math change without a before/after PROFILE_STAGES
diff **and** a garden PSNR/SSIM regression check (> −0.05 dB needs justification).

## Where the time goes (garden, `-n 1500 -d 2`, M4/16GB)

From `BENCHMARK=1 PROFILE_STAGES=1` (see [benchmark ledger](../benchmarks/)):

| Stage | share | implication |
|---|---|---|
| `rasterize_backward` | **~54%** | the elephant — top lever |
| `rasterize_forward` | ~29% | second |
| `project_and_sh_backward_adam` | ~9% | next (sparse Adam targets this) |
| `project_and_sh_forward` | ~4% | |
| `prefix_sort_pack` | ~3% | not worth optimizing |
| `ssim_fwd_bwd` | ~1% | not worth optimizing |
| `grad_stats` | ~1% | |

**Rasterization fwd+bwd ≈ 82%.** This reorders the original todo: the prefix-sum
collapse and SSIM fusion target stages already <3% and are no longer worth the
risk. Focus on the rasterizer and the fused Adam.

## Speed (M3) — ordered by profit ÷ risk

### M3.1 Sparse / visible-only Adam — *already largely realized*
**Finding (2026-06-10):** the expensive fused `project_and_sh_backward_kernel`
already early-returns on `radii[idx] <= 0` (msplat_metal.metal:1846), so the SH
backward + its fused Adam update — the ~9% `proj_sh_bwd_adam` stage — already skips
invisible gaussians. The remaining four parameter groups use the element-wise
`fused_adam_kernel` (means/scales/quats/opacity), which runs over all params
unconditionally but is *not* a top profile stage (negligible). So Taming-style
sparse Adam is effectively in place; the only addressable remainder (gating the
four element-wise groups) targets a sub-1% cost and isn't worth the complexity.
- Reference: Taming 3DGS (arXiv:2406.15643); inria `--optimizer_type sparse_adam`.

### M3.3 Warp/simdgroup-aggregated atomics in `rasterize_backward` — *biggest, hardest* (~54%)
Today the backward issues atomics per warp-leader across 4 warps. Do a full
simdgroup reduction, then **one** `atomic_fetch_add` per simdgroup per parameter,
cutting atomic traffic. High value, high risk (correctness of the reduction) —
prototype behind a function constant and A/B against the current path on garden.

### M3.6 Densification buffer reuse (memory + latency)
`ensureCapacity` allocates new backing buffers before freeing the old (the source
of the inflated "footprint" in the 7K run). Pre-size to the expected post-refine
count, or double-buffer and reuse, to cut transient allocation and refine latency.

### Already done / dropped
- **Async loss readback** (old M3.2): already the case — `msplat_train_step` reads
  `loss_sum` without a sync. No work.
- **Prefix-sum collapse / SSIM fusion** (old M3.4/M3.5): deprioritized — those
  stages are <3% each; not worth the regression risk.
- **Radix vs bitonic sort:** sort+pack is ~3%; leave the tile-local bitonic.

## M4.0 — over-downscale instability on small-image datasets (found + fixed 2026-06-10)

A 13-scene sweep (`tested_outputs/RESULTS.md`) at a uniform `-d4` showed all 7
Mip-NeRF scenes fine (PSNR 24–30) but all 4 Tanks&Temples/Deep Blending at PSNR 4–6.
**Root cause (controlled ablation, only `-d` changed): over-downscaling.** Those
datasets ship ~979px images; `-d4` renders at 244px (early progressive steps ~61px)
and the optimizer destabilizes. At native `-d1` all 4 train well (truck 23.5,
drjohnson/playroom ~28). The `loss=nan` first observed was a *late symptom*; the
initial "scale explosion" guess was overturned by the `-d1` recovery.

Shipped this session: **coarse-render warning** (`camera_math.hpp::isCoarseRender`,
CLI warns < 400px long side, unit-tested) + **NaN/Inf guard** in Adam (safety net;
makes outputs finite but does not by itself rescue an over-downscaled run).

Deeper (future, needs multi-scene validation): msplat should be *stable* (just
lower quality) at coarse res like reference 3DGS — candidate hardening is a scale
clamp / gradient clip / scale-reg (3DGS-MCMC ≈0.01). Practical guidance until then:
pick `-d` by native resolution (target ~1 MP render), not a fixed factor; gsplata
should do the same.

## Quality (M4) — close the garden SSIM gap, fewer splats

> Note: hold the schedule (`--num-downscales`) and background fixed across A/B runs.
> Measured: black vs magenta backgrounds are within noise on garden (24.98 vs 24.95);
> the bigger lever on the headline number is `--num-downscales` (see the ledger).

### M4.1 Anti-aliasing (Mip-Splatting 2D filter) — cheap, in-projection
Today 3DGS adds a fixed `0.3` to the 2D covariance diagonal with no opacity
correction. The antialiased variant convolves with a screen-space Gaussian of
variance `s = 0.3` **and compensates opacity**:

```
Σ'_2D = Σ_2D + s·I                                   (s = 0.3, = gsplat eps2d)
ρ      = sqrt( det(Σ_2D) / det(Σ_2D + s·I) )         opacity *= ρ
```

`ρ → 1` for large splats, `< 1` for sub-pixel splats (so tiny gaussians become
translucent instead of opaque dots). Plug into projection after computing the 2D
covariance; propagate gradients through `ρ` in the backward. Targets the SSIM gap.
- Sources: Mip-Splatting (CVPR 2024, arXiv:2311.16493); gsplat
  `rasterize_mode="antialiased"`, `eps2d=0.3`.
- Note: the paper also adds a separate 3D smoothing filter (training-time); gsplat's
  `antialiased` flag implements only the 2D compensation above.

### M4.2 AbsGS — absolute-gradient densification
Standard densification sums signed per-pixel `∂L/∂μ_2d` before taking the norm, so
opposite-sign contributions cancel and detailed gaussians never split. Accumulate
**absolute** per-component contributions in `accumulate_grad_stats` (a separate
"absgrad" accumulator; the optimizer's signed gradient is unchanged):

```
ĝ_x = Σ_pixels |∂L/∂μ_x|,  ĝ_y = Σ_pixels |∂L/∂μ_y|;  densify if ‖(ĝ_x, ĝ_y)‖ > τ
```

Raise the threshold from `0.0002`. **Sources disagree:** AbsGS paper recommends
`τ ≈ 0.0004`; gsplat docs lean `0.0006–0.0008`. Make it a CLI flag and sweep.
- Sources: AbsGS (arXiv:2404.10484, Eqs. 5, 12–13); gsplat `absgrad=True`.

### M4.3 MCMC relocation (`--strategy mcmc`) — fixed-budget, pairs with `--max-splats`
Replace clone/split with relocation of dead gaussians (opacity ≤ 0.005) onto
opacity-sampled survivors, plus opacity-gated noise on means. Relocation preserves
the render: `o_new = 1 − (1 − o_old)^{1/N}`, covariance rescaled accordingly; noise
`∝ σ(−k(o−t))·Σ·η`, defaults `k=100, t=0.995`. Growth holds at `cap_max` — a true
fixed budget (natural fit for `--max-splats`). Larger lift; prototype in the Python
bindings first.
- Sources: 3DGS-MCMC (NeurIPS 2024); official `ubc-vision/3dgs-mcmc` `train.py`,
  `arguments/__init__.py` (note: code `noise_lr = 5e5`, effective via `xyz_lr`).

### M4.4 Opacity decay
Per-step opacity decay (Brush uses `--opac-decay 0.004`) as a smoother alternative
to the hard reset every 30 refines. Evaluate on ≥3 scenes.

## Borrowing from Brush / OpenSplat

- **OpenSplat** progress pattern (one newline-terminated `Step N: loss (X%)` line,
  no `\r`, flush on newline) — msplat's `--progress-every` follows the same
  pipeline-safe shape (we add splats/ms + a jsonl variant).
- **Brush** `--max-splats` (default 10M) as the budget knob (we implemented a cap);
  MCMC `--mean-noise-weight` and `--opac-decay` inform M4.3/M4.4; templated export
  names (`export_{iter}.ply`) are a nice pattern for `--save-every`.
- Reference implementations to port math from: **gsplat** (AA/absgrad/MCMC/bilateral
  grid), **Taming-3DGS** (fused SSIM — partly done — and sparse Adam),
  **fused-ssim** (rahul-goel) for the SSIM kernel.

## Process

1. Profile first; name the stage and its share.
2. Prefer removing dispatches/syncs/round-trips over ALU micro-tuning.
3. Report PSNR/SSIM with every perf change; record in the ledger + CHANGELOG.
4. Land only what validates without quality regression; otherwise leave it
   specified here as ready-to-implement future work.
