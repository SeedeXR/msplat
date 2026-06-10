# Datasets

Training scenes are hosted on Hugging Face, **not** committed to this repo:
**[alexmkwizu/gaussian_training_datasets](https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets)**

The repo ships only a small `garden` scene (`datasets/mipnerf360/garden`, LFS) for
quickstart and CI. Everything else under `datasets/` is git-ignored — pulled from
Hugging Face and cached locally, never committed here.

## Layout

The Hugging Face repo mirrors the local `datasets/` layout (COLMAP format):

```
datasets/
  mipnerf360/{bicycle,bonsai,counter,garden,kitchen,room,stump}/
      images/            ← training images
      sparse/0/          ← cameras.bin, images.bin, points3D.bin
  tandt/{train,truck}/   ← Tanks & Temples
  db/{drjohnson,playroom}/  ← Deep Blending
```

## Download

```bash
pip install -U "huggingface_hub[cli]"

# Everything into ./datasets/
huggingface-cli download alexmkwizu/gaussian_training_datasets \
    --repo-type dataset --local-dir datasets

# Or just one scene
huggingface-cli download alexmkwizu/gaussian_training_datasets \
    --repo-type dataset --include "mipnerf360/room/*" --local-dir datasets
```

(`git clone https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets datasets`
also works if you have git-lfs.)

## Choosing `--downscale-factor` (important)

Over-downscaling **small** images destabilizes training (the optimizer can diverge,
not just lose quality — see [optimization-roadmap M4.0](dev/optimization-roadmap.md)).
Pick `-d` by the *native* resolution, targeting a ~1 MP render:

| dataset | native image | use | render |
|---|---|---|---|
| Mip-NeRF 360 | ~16 MP (e.g. 4946×3286) | `-d 4` | ~1 MP |
| Tanks & Temples | ~1 MP (e.g. 979×546) | `-d 1` | native |
| Deep Blending | ~1–2 MP | `-d 1` | native |

The CLI prints a warning when the render long-side falls below ~400 px. On a 16 GB
machine, full-res Mip-NeRF (`-d 1`) OOMs (all images are decoded up-front); `-d 4`
is the right setting there. The small-image datasets fit comfortably at `-d 1`.

```bash
msplat datasets/mipnerf360/garden -n 7000 -d 4 --eval
msplat datasets/tandt/truck      -n 7000 -d 1 --eval
```

## Adding a new dataset and pushing it to Hugging Face

1. Place the scene under `datasets/<group>/<scene>/` in COLMAP layout
   (`images/` + `sparse/0/{cameras,images,points3D}.bin`). Confirm msplat reads it:
   ```bash
   msplat datasets/<group>/<scene> -n 50 -d 1 --eval   # quick sanity
   ```
2. Authenticate once (needs write access to the dataset repo):
   ```bash
   huggingface-cli login
   ```
3. Upload it (path-in-repo mirrors the local path):
   ```bash
   huggingface-cli upload alexmkwizu/gaussian_training_datasets \
       datasets/<group>/<scene> <group>/<scene> --repo-type dataset
   ```

Do **not** `git add` datasets into the msplat repo — they are git-ignored on purpose
and live on Hugging Face. (The only exception is the bundled `garden` scene; leave it
as committed for CI and don't replace it with full-resolution images.)
