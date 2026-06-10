#ifndef MSPLAT_CAMERA_MATH_HPP
#define MSPLAT_CAMERA_MATH_HPP

// Pure (no-GPU) camera/resolution math, shared by the loader and exercised by the
// C++ unit tests. Keeping it here (not inline in input_data.cpp) makes the COLMAP
// "declared dims != actual image dims" rescale — and the over-downscale guard —
// regression-testable without a GPU or a real image on disk.

namespace msplat {

struct CamIntrinsics { float fx, fy, cx, cy; };

// COLMAP records intrinsics for the resolution at which SfM ran. Datasets often
// ship images downscaled from that (e.g. Tanks&Temples: cameras.bin says 1957x1091
// but images are 979x546). Rescale the intrinsics to the ACTUAL image resolution so
// projection is correct. Identity when dims match or declared dims are unknown (0).
inline CamIntrinsics rescaleIntrinsics(CamIntrinsics in,
                                       int declaredW, int declaredH,
                                       int actualW, int actualH) {
    if (declaredW <= 0 || declaredH <= 0) return in;          // no declared dims
    if (actualW == declaredW && actualH == declaredH) return in;
    float sx = (float)actualW / (float)declaredW;
    float sy = (float)actualH / (float)declaredH;
    return { in.fx * sx, in.fy * sy, in.cx * sx, in.cy * sy };
}

// Render resolution after an integer-ish downscale factor (matches the loader's
// width/downscale truncation).
inline int downscaledExtent(int extent, float downscale) {
    if (downscale <= 1.0f) return extent;
    return (int)(extent / downscale);
}

// Training becomes numerically unstable (the optimizer diverges, not just lower
// quality) when the rendered image is too small — observed on Tanks&Temples/Deep
// Blending whose ~979px images at -d4 render at 244px and diverge. Long side below
// this is the footgun; warn and suggest a smaller -d. (Note: progressive
// --num-downscales makes the EARLY steps coarser still.)
inline constexpr int COARSE_RENDER_WARN_PX = 400;

inline bool isCoarseRender(int renderW, int renderH) {
    int longSide = renderW > renderH ? renderW : renderH;
    return longSide > 0 && longSide < COARSE_RENDER_WARN_PX;
}

}  // namespace msplat

#endif
