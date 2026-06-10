// Pure C++ unit tests — no GPU, no dataset. Exercise the shared logic helpers
// (cli_support.hpp, ply_header.hpp) so format/schedule/parse regressions are
// caught in CI before the slower GPU integration tests run.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "cli_support.hpp"
#include "ply_header.hpp"

#include <sstream>

using namespace msplat;

TEST_CASE("numShBasesForDegree matches (deg+1)^2 and clamps") {
    CHECK(numShBasesForDegree(0) == 1);
    CHECK(numShBasesForDegree(1) == 4);
    CHECK(numShBasesForDegree(2) == 9);
    CHECK(numShBasesForDegree(3) == 16);
    CHECK(numShBasesForDegree(4) == 25);
    CHECK(numShBasesForDegree(-3) == 1);   // clamp low
    CHECK(numShBasesForDegree(9) == 25);   // clamp high
}

TEST_CASE("downscaleFactorForStep follows the progressive schedule") {
    // numDownscales=2, schedule=3000 -> 4x until 3000, 2x until 6000, then 1x.
    CHECK(downscaleFactorForStep(0,    2, 3000) == 4);
    CHECK(downscaleFactorForStep(2999, 2, 3000) == 4);
    CHECK(downscaleFactorForStep(3000, 2, 3000) == 2);
    CHECK(downscaleFactorForStep(6000, 2, 3000) == 1);
    CHECK(downscaleFactorForStep(100000, 2, 3000) == 1);  // floored at 1
    CHECK(downscaleFactorForStep(0,    0, 3000) == 1);     // no downscales
    CHECK(downscaleFactorForStep(10,   2, 0)    == 4);     // guard schedule<=0
}

TEST_CASE("formatProgress plain form is exactly what the pipeline parses") {
    std::string s = formatProgress(/*jsonl=*/false, 1000, 30000, 250000, 0.0321f, 7.42);
    CHECK(s == "step=1000/30000 splats=250000 loss=0.032100 ms/it=7.42");
}

TEST_CASE("formatProgress jsonl form is valid single-line JSON") {
    std::string s = formatProgress(/*jsonl=*/true, 1000, 30000, 250000, 0.0321f, 7.42);
    CHECK(s == "{\"step\":1000,\"total\":30000,\"splats\":250000,\"loss\":0.032100,\"ms_per_step\":7.420}");
    CHECK(s.find('\n') == std::string::npos);   // must be single-line
}

TEST_CASE("maxSplatsFromBudgetGB is zero-safe and monotonic") {
    CHECK(maxSplatsFromBudgetGB(0.0) == 0);
    CHECK(maxSplatsFromBudgetGB(-5.0) == 0);
    long long b8  = maxSplatsFromBudgetGB(8.0);
    long long b16 = maxSplatsFromBudgetGB(16.0);
    CHECK(b8 > 0);
    CHECK(b16 == doctest::Approx((double)b8 * 2).epsilon(0.001));
    // 16 GB -> ~9.8M splat ceiling at 900 B/splat, 55% usable.
    CHECK(b16 > 9'000'000);
    CHECK(b16 < 11'000'000);
}

// Build a minimal 3DGS PLY header the way save_gaussians.cpp does.
static std::string makeHeader(int step, long n, int numDc, int numFr) {
    std::ostringstream o;
    o << "ply\nformat binary_little_endian 1.0\n";
    o << plyStepCommentPrefix() << step << "\n";
    o << "element vertex " << n << "\n";
    o << "property float x\nproperty float y\nproperty float z\n";
    o << "property float nx\nproperty float ny\nproperty float nz\n";
    for (int i = 0; i < numDc; i++) o << "property float f_dc_" << i << "\n";
    for (int i = 0; i < numFr; i++) o << "property float f_rest_" << i << "\n";
    o << "property float opacity\n";
    o << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    o << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    o << "end_header\n";
    return o.str();
}

TEST_CASE("parsePlyHeader round-trips writer output (guards the --resume step bug)") {
    std::istringstream f(makeHeader(/*step=*/12345, /*n=*/987654, /*numDc=*/3, /*numFr=*/45));
    PlyHeaderInfo h = parsePlyHeader(f);
    CHECK(h.isPly);
    CHECK(h.step == 12345);          // the exact regression that broke resume
    CHECK(h.numPoints == 987654);
    CHECK(h.numDc == 3);
    CHECK(h.numFr == 45);
}

TEST_CASE("parsePlyHeader handles missing step comment (older PLYs)") {
    std::istringstream f(
        "ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
        "property float x\nproperty float f_dc_0\nend_header\n");
    PlyHeaderInfo h = parsePlyHeader(f);
    CHECK(h.isPly);
    CHECK(h.step == 0);              // absent -> 0, not a crash
    CHECK(h.numPoints == 10);
    CHECK(h.numDc == 1);
}

TEST_CASE("parsePlyHeader rejects non-PLY input") {
    std::istringstream f("not a ply file\nrandom bytes\n");
    PlyHeaderInfo h = parsePlyHeader(f);
    CHECK_FALSE(h.isPly);
}
