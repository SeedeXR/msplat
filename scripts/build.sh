#!/usr/bin/env bash
# msplat build script — configures and builds the C++ CLI + core lib + Metal
# kernels + C++ unit tests via CMake, with a preflight check for the Metal
# toolchain (a separately-downloadable Xcode component on recent Xcode).
#
# Usage:
#   scripts/build.sh                 # Release build + run unit tests
#   scripts/build.sh --debug         # Debug build
#   scripts/build.sh --clean         # wipe build/ first
#   scripts/build.sh --no-tests      # skip building/running unit tests
#   scripts/build.sh --python        # also build the Python extension (needs nanobind)
#   scripts/build.sh --jobs N        # parallelism (default: sysctl hw.ncpu)
#   scripts/build.sh --build-dir DIR # build directory (default: build)
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

BUILD_TYPE=Release
BUILD_DIR=build
RUN_TESTS=1
BUILD_PYTHON=OFF
CLEAN=0
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      BUILD_TYPE=Debug; shift ;;
        --release)    BUILD_TYPE=Release; shift ;;
        --clean)      CLEAN=1; shift ;;
        --no-tests)   RUN_TESTS=0; shift ;;
        --python)     BUILD_PYTHON=ON; shift ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ── Preflight ──────────────────────────────────────────────────────────────
command -v cmake >/dev/null || { echo "error: cmake not found (brew install cmake)" >&2; exit 1; }
command -v xcrun >/dev/null || { echo "error: Xcode command line tools not found" >&2; exit 1; }

if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    echo "error: the Metal toolchain is not installed." >&2
    echo "       Install it once with:" >&2
    echo "         xcodebuild -downloadComponent MetalToolchain" >&2
    echo "       (requires full Xcode; ~690 MB download)." >&2
    exit 1
fi

ARCH="$(uname -m)"
[[ "$ARCH" == "arm64" ]] || echo "warning: msplat targets Apple Silicon (arm64); detected $ARCH." >&2

echo "==> msplat build  type=$BUILD_TYPE  jobs=$JOBS  python=$BUILD_PYTHON  dir=$BUILD_DIR"
[[ "$CLEAN" == "1" ]] && { echo "==> cleaning $BUILD_DIR"; rm -rf "$BUILD_DIR"; }

# ── Configure + build ────────────────────────────────────────────────────────
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DMSPLAT_BUILD_TESTS=$([[ "$RUN_TESTS" == "1" ]] && echo ON || echo OFF) \
    -DMSPLAT_BUILD_PYTHON="$BUILD_PYTHON"

cmake --build "$BUILD_DIR" -j "$JOBS"

# ── Unit tests ───────────────────────────────────────────────────────────────
if [[ "$RUN_TESTS" == "1" ]]; then
    echo "==> running unit tests"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo ""
echo "==> done. Artifacts:"
echo "    CLI:      $BUILD_DIR/msplat"
echo "    metallib: $BUILD_DIR/default.metallib"
echo "    lib:      $BUILD_DIR/libmsplat_core.a"
"$BUILD_DIR/msplat" --version 2>/dev/null | head -1 | sed 's/^/    version:  /' || true
