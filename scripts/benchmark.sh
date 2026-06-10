#!/usr/bin/env bash
# Multi-scene benchmark / regression harness for msplat.
#
# For each scene it runs a quick CHUNK sanity train, then a full(er) WHOLE train,
# captures PSNR/SSIM/L1/gaussians/wall-time + peak resident memory, writes the
# final PLY to the output dir, and appends a row to <out>/SUMMARY.md.
#
# A "scene" is any directory msplat can auto-detect: it has sparse/0/cameras.bin
# (COLMAP), transforms.json (Nerfstudio), or cameras.json / keyframes/ (Polycam).
#
# Usage:
#   scripts/benchmark.sh [options] <scene_or_parent_dir> [more dirs...]
# Options:
#   --iters N        whole-run iterations (default 7000)
#   --downscale D    whole-run downscale factor (default 1 = full res)
#   --chunk N        chunk-run iterations (default 200)
#   --chunk-d D      chunk-run downscale (default 4)
#   --out DIR        output dir for PLYs + SUMMARY.md (default tested_outputs)
#   --bin PATH       msplat binary (default build/msplat)
#   --max-splats N   pass --max-splats to the whole run (default: unset)
#   --whole-only     skip the chunk runs
set -uo pipefail
cd "$(dirname "$0")/.."

ITERS=7000 DOWNSCALE=1 CHUNK=200 CHUNK_D=4
OUT=tested_outputs BIN=build/msplat MAXSPLATS="" WHOLE_ONLY=0
SCENES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters) ITERS="$2"; shift 2 ;;
    --downscale) DOWNSCALE="$2"; shift 2 ;;
    --chunk) CHUNK="$2"; shift 2 ;;
    --chunk-d) CHUNK_D="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --max-splats) MAXSPLATS="$2"; shift 2 ;;
    --whole-only) WHOLE_ONLY=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) SCENES+=("$1"); shift ;;
  esac
done
[[ ${#SCENES[@]} -gt 0 ]] || { echo "error: give at least one scene or parent dir" >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "error: binary not found: $BIN (run scripts/build.sh)" >&2; exit 1; }

is_scene() {  # $1 = dir
  [[ -f "$1/sparse/0/cameras.bin" || -f "$1/sparse/0/cameras.txt" \
     || -f "$1/cameras.bin" || -f "$1/transforms.json" \
     || -f "$1/cameras.json" || -d "$1/keyframes/corrected_cameras" ]]
}

# Expand parent dirs into scene roots.
ROOTS=()
for s in "${SCENES[@]}"; do
  if is_scene "$s"; then ROOTS+=("$s")
  else
    for sub in "$s"/*/; do sub="${sub%/}"; is_scene "$sub" && ROOTS+=("$sub"); done
  fi
done
[[ ${#ROOTS[@]} -gt 0 ]] || { echo "error: no msplat-readable scenes found under: ${SCENES[*]}" >&2; exit 3; }

mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"
{
  echo "# msplat multi-scene results"
  echo ""
  echo "machine: $(sysctl -n hw.model 2>/dev/null) · $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1073741824" GB"}') · $($BIN --version 2>/dev/null | head -1)"
  echo "whole run: -n $ITERS -d $DOWNSCALE${MAXSPLATS:+ --max-splats $MAXSPLATS} --eval   |   chunk: -n $CHUNK -d $CHUNK_D"
  echo ""
  echo "| scene | chunk PSNR | whole PSNR | whole SSIM | L1 | gaussians | wall (s) | peak RSS (GB) | PLY |"
  echo "|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

run_eval() {  # $1=root $2=iters $3=downscale $4=out_ply($5 extra) -> echoes "PSNR SSIM L1 GAUSS"
  local root="$1" iters="$2" d="$3" ply="$4"; shift 4
  local log; log="$(mktemp)"
  local ms_args=()
  [[ -n "$MAXSPLATS" ]] && ms_args=(--max-splats "$MAXSPLATS")   # array → two words, not one
  /usr/bin/time -l "$BIN" "$root" -n "$iters" -d "$d" --eval --progress-every 1000 \
      "${ms_args[@]}" "$@" -o "$ply" >"$log" 2>"$log.time"
  local code=$?
  local line; line="$(grep -E '^  PSNR:' "$log" | tail -1)"
  local psnr ssim l1 gauss
  psnr=$(awk '{for(i=1;i<=NF;i++)if($i=="PSNR:")print $(i+1)}' <<<"$line")
  ssim=$(awk '{for(i=1;i<=NF;i++)if($i=="SSIM:")print $(i+1)}' <<<"$line")
  l1=$(awk   '{for(i=1;i<=NF;i++)if($i=="L1:")print $(i+1)}' <<<"$line")
  gauss=$(awk '{for(i=1;i<=NF;i++)if($i=="Gaussians:")print $(i+1)}' <<<"$line")
  local rss; rss=$(awk '/maximum resident set size/{printf "%.2f", $1/1073741824}' "$log.time")
  local wall; wall=$(awk '/real/{print $1}' "$log.time" | head -1)
  echo "$code|${psnr:-NA}|${ssim:-NA}|${l1:-NA}|${gauss:-NA}|${rss:-NA}|${wall:-NA}"
  rm -f "$log" "$log.time"
}

for root in "${ROOTS[@]}"; do
  name="$(basename "$root")"
  echo "==> $name"
  chunk_psnr="—"
  if [[ "$WHOLE_ONLY" -eq 0 ]]; then
    echo "    chunk (-n $CHUNK -d $CHUNK_D)…"
    IFS='|' read -r ccode cpsnr _ _ _ _ _ < <(run_eval "$root" "$CHUNK" "$CHUNK_D" "$OUT/${name}_chunk.ply")
    [[ "$ccode" == "0" ]] && chunk_psnr="$cpsnr" || chunk_psnr="FAIL($ccode)"
    rm -f "$OUT/${name}_chunk.ply"   # keep only the whole-run PLY
  fi
  echo "    whole (-n $ITERS -d $DOWNSCALE)…"
  IFS='|' read -r wcode wpsnr wssim wl1 wg wrss wwall < <(run_eval "$root" "$ITERS" "$DOWNSCALE" "$OUT/${name}.ply")
  if [[ "$wcode" != "0" ]]; then
    echo "| $name | $chunk_psnr | FAIL($wcode) | — | — | — | — | — | — |" >> "$SUMMARY"
    echo "    FAILED (exit $wcode)"; continue
  fi
  echo "| $name | $chunk_psnr | $wpsnr | $wssim | $wl1 | $wg | $wwall | $wrss | ${name}.ply |" >> "$SUMMARY"
  echo "    PSNR=$wpsnr SSIM=$wssim gaussians=$wg wall=${wwall}s rss=${wrss}GB"
done

echo ""; echo "Summary written to $SUMMARY"; cat "$SUMMARY"
