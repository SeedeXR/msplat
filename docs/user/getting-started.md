# Getting started

msplat trains a 3D Gaussian Splatting scene from a photogrammetry capture entirely
on the Apple GPU (Metal) — no CUDA, no PyTorch. It runs on any Apple Silicon Mac
(macOS 14+), including a base M4 / 16 GB MacBook Pro.

## Install

### Python (easiest)
```bash
pip install msplat            # library
pip install msplat[cli]       # + the msplat-train command
```

### Build the C++ CLI from source
Requires full Xcode and the Metal toolchain (a one-time download on recent Xcode):
```bash
xcodebuild -downloadComponent MetalToolchain     # once, ~690 MB
git clone https://github.com/SeedeXR/msplat && cd msplat
./scripts/build.sh                                # Release build + unit tests
./build/msplat --version
```
`scripts/build.sh --help` lists options (`--debug`, `--clean`, `--python`, `--jobs N`).

## Train your first scene

```bash
msplat /path/to/scene -n 7000 --eval
```
- `/path/to/scene` is the **project root** — the folder that *contains* `sparse/`
  and `images/` (not `sparse/0/`).
- `--eval` holds out every 8th view and prints PSNR/SSIM/L1 at the end.
- Output PLY defaults to `splat.ply`; set it with `-o /path/out.ply`.

Watch progress stream as it trains (every 100 steps by default):
```
step=1000/7000 splats=289114 loss=0.054313 ms/it=10.71
...
Done: 7000 iters, 755235 Gaussians, PSNR 24.98, wrote /abs/path/splat.ply
```

## Getting training scenes

A small `garden` scene ships in-repo. For the full set (Mip-NeRF 360, Tanks &
Temples, Deep Blending), create a `datasets/` folder and download from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
hf download alexmkwizu/gaussian_training_datasets \
    --repo-type dataset --local-dir datasets
```

See **[datasets.md](../datasets.md)** for single-scene downloads, the per-dataset
`--downscale-factor` guidance (small-image datasets train at `-d 1`, not `-d 4`),
and how to push your own scenes to the Hugging Face repo.

## Dataset formats (auto-detected)

| Format | What msplat looks for |
|---|---|
| **COLMAP** | `sparse/0/{cameras,images,points3D}.bin` (or `.txt`) + `images/` |
| **Nerfstudio** | `transforms.json` (+ images, optional point cloud) |
| **Polycam** | `keyframes/corrected_cameras/` + `corrected_images/`, or `cameras.json` |

Pinhole / simple-radial / radial / OpenCV cameras are supported; Brown-Conrady
distortion is undistorted automatically on load.

## Quality presets

```bash
msplat /path/to/scene --preset draft        # fast: 7000 iters, half-res, SH2, ≤1M splats
msplat /path/to/scene --preset balanced     # 30000 iters, ≤3M splats
msplat /path/to/scene --preset production   # 100000 iters, ≤6M splats
```
Any flag you add after a preset overrides it (e.g. `--preset draft -n 3000`).

## Running on a 16 GB machine alongside other apps

msplat is designed to be a good citizen on a memory-constrained Mac:
```bash
msplat /path/to/scene --preset balanced --max-splats 2000000
# or let a budget derive the cap:
msplat /path/to/scene --memory-budget 10
```
- `--max-splats N` caps the gaussian count; above the cap the trainer prunes but
  stops growing, bounding peak memory. A full-res garden scene at 7K trains in
  ~4.9 GB resident on a 16 GB M4 (see [benchmarks](../benchmarks/)).
- `--memory-budget GB` derives a cap from a memory budget.

## Resume and checkpoints

```bash
msplat /path/to/scene -n 30000 --save-every 5000     # periodic *_5000.ply checkpoints
msplat /path/to/scene -n 30000 --resume splat_5000.ply  # continues the schedule
```

## Pause / cancel

- `Ctrl-C` (SIGINT) or SIGTERM: msplat finishes the current iteration, writes
  `<output>_interrupted.ply`, and exits (130 / 143).
- SIGSTOP / SIGCONT pause and resume (handled by the OS).

## Background color

The default training background is **black**. To highlight under-reconstructed
regions during debugging, use `--debug-bg` (magenta), or set any color with
`--bg-color R G B` (each 0–1).

## Output

A binary 3DGS PLY (`x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*`,
log-space scales, pre-sigmoid opacity) plus a `cameras.json`. Use the `.splat`
extension on `-o` to write the compact splat format instead. The PLY opens in any
standard 3DGS viewer.

See `man docs/msplat.1` for the complete flag reference.
