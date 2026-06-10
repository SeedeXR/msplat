#ifndef MSPLAT_CLI_SUPPORT_HPP
#define MSPLAT_CLI_SUPPORT_HPP

// Pure, dependency-free helpers shared by the CLI and the model and exercised by
// the C++ unit tests (tests/cpp). Nothing here may touch Metal/GPU state so the
// unit-test target can link and run without a device or a dataset.

#include <string>
#include <cstdio>
#include <cstdint>
#include <algorithm>

namespace msplat {

// Number of spherical-harmonics basis functions for a given max degree (0..4).
// (degree+1)^2, clamped: deg0=1, deg1=4, deg2=9, deg3=16, deg4=25.
inline int numShBasesForDegree(int degree) {
    if (degree < 0) degree = 0;
    if (degree > 4) degree = 4;
    return (degree + 1) * (degree + 1);
}

// Progressive-resolution schedule: the image is downscaled by 2^k where k starts
// at numDownscales and drops by one every resolutionSchedule steps, floored at 0.
// Mirrors Model::getDownscaleFactor — kept here as the single tested source.
inline int downscaleFactorForStep(int step, int numDownscales, int resolutionSchedule) {
    if (resolutionSchedule <= 0) return 1 << std::max(numDownscales, 0);
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

// One progress line. plain form is what gsplata's parser keys on; jsonl is the
// machine-stable variant. Both are single-line and newline-terminated.
inline std::string formatProgress(bool jsonl, std::size_t step, int total,
                                  long splats, float loss, double msPerStep) {
    char buf[256];
    if (jsonl)
        std::snprintf(buf, sizeof(buf),
            "{\"step\":%zu,\"total\":%d,\"splats\":%ld,\"loss\":%.6f,\"ms_per_step\":%.3f}",
            step, total, splats, loss, msPerStep);
    else
        std::snprintf(buf, sizeof(buf),
            "step=%zu/%d splats=%ld loss=%.6f ms/it=%.2f",
            step, total, splats, loss, msPerStep);
    return std::string(buf);
}

// Advisory splat cap from a unified-memory budget (GB). ~900 B per splat covers
// params + 2x Adam moments + densify scratch + sort bins; reserve ~45% of the
// budget for decoded images, framebuffers and the OS. Returns 0 for a 0 budget.
inline long long maxSplatsFromBudgetGB(double budgetGB) {
    if (budgetGB <= 0.0) return 0;
    double usableBytes = budgetGB * 1e9 * 0.55;
    return (long long)(usableBytes / 900.0);
}

}  // namespace msplat

#endif
