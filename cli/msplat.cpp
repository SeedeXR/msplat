#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <climits>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "random_iter.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "bindings.h"
#include "cli_support.hpp"
#include "camera_math.hpp"

namespace fs = std::filesystem;

// ── Exit codes (documented in docs/msplat.1) ────────────────────────────────
// 0 success · 2 bad args (CLI11) · 3 dataset load failure · 4 GPU/Metal init
// failure (reserved) · 5 output write failure · 130 SIGINT · 143 SIGTERM.
enum ExitCode { EXIT_OK = 0, EXIT_LOAD = 3, EXIT_GPU = 4, EXIT_WRITE = 5 };

// ── Cooperative signal handling ─────────────────────────────────────────────
// SIGINT/SIGTERM set a flag; the training loop finishes its in-flight iteration,
// writes a partial checkpoint, and exits with 128+signum. Async-signal-safe:
// the handler only does an atomic store.
static std::atomic<int> g_caught_signal{0};
static void msplat_signal_handler(int sig) { g_caught_signal.store(sig, std::memory_order_relaxed); }
static void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = msplat_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: let blocking calls return so we can stop promptly
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

int main(int argc, char *argv[]) {
    CLI::App app{"msplat — 3D Gaussian Splatting for Apple Silicon"};
    app.set_version_flag("--version", std::string(APP_VERSION) + "\nmetallib: built-in (default.metallib beside binary)");

    // Required
    std::string projectRoot;
    app.add_option("input", projectRoot, "Path to dataset (COLMAP, Nerfstudio, Polycam)")
        ->required()
        ->check(CLI::ExistingDirectory);

    // Output
    std::string outputScene = "splat.ply";
    app.add_option("-o,--output", outputScene, "Output scene path");
    int saveEvery = -1;
    app.add_option("-s,--save-every", saveEvery, "Save every N steps (-1 to disable)");

    // Resume
    std::string resume;
    app.add_option("--resume", resume, "Resume training from PLY file")
        ->check(CLI::ExistingFile);

    // Validation
    bool validate = false;
    app.add_flag("--val", validate, "Withhold a camera for validation");
    std::string valImage = "random";
    app.add_option("--val-image", valImage, "Validation image filename");
    std::string valRender;
    app.add_option("--val-render", valRender, "Directory to render validation images");

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(2, 100));

    // Training hyperparameters
    int numIters = 30000;
    auto *optIters = app.add_option("-n,--num-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000))->group("Training");
    float downScaleFactor = 1.0f;
    auto *optDownscale = app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f))->group("Training");
    int numDownscales = 2;
    app.add_option("--num-downscales", numDownscales, "Progressive downscale levels");
    int resolutionSchedule = 3000;
    app.add_option("--resolution-schedule", resolutionSchedule, "Double resolution every N steps");
    int shDegree = 3;
    auto *shDegreeOpt = app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4))->group("Training");
    int shDegreeInterval = 1000;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps");
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    int refineEvery = 100;
    app.add_option("--refine-every", refineEvery, "Densify/prune every N steps");
    int warmupLength = 500;
    app.add_option("--warmup-length", warmupLength, "Steps before first densification");
    int resetAlphaEvery = 30;
    app.add_option("--reset-alpha-every", resetAlphaEvery, "Reset opacity every N refinements");
    float densifyGradThresh = 0.0002f;
    app.add_option("--densify-grad-thresh", densifyGradThresh, "Gradient threshold for split/dup");
    float densifySizeThresh = 0.01f;
    app.add_option("--densify-size-thresh", densifySizeThresh, "Size threshold (dup vs split)");
    int stopScreenSizeAt = 4000;
    app.add_option("--stop-screen-size-at", stopScreenSizeAt, "Stop splitting large gaussians after N steps");
    float splitScreenSize = 0.05f;
    app.add_option("--split-screen-size", splitScreenSize, "Screen-space split threshold");
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    std::vector<float> bgColor = {0.0f, 0.0f, 0.0f};   // default black (pipeline-safe)
    auto *optBgColor = app.add_option("--bg-color", bgColor, "Background RGB (0-1), default black")
        ->expected(3)->group("Output");
    bool debugBg = false;
    app.add_flag("--debug-bg", debugBg, "Use magenta debug background (highlights gaps); ignored if --bg-color is set")
        ->group("Output");
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");

    // ── Quality presets (M5.3) ── applied only to flags the user did NOT set,
    // so any explicit flag overrides the preset regardless of order.
    std::string preset;
    app.add_option("--preset", preset, "Quality preset: draft | balanced | production")
        ->check(CLI::IsMember({"draft", "balanced", "production"}))->group("Training");

    // ── Resource governance (M2) ──
    int64_t maxSplats = 0;
    auto *optMaxSplats = app.add_option("--max-splats", maxSplats,
        "Hard cap on gaussian count (0 = unlimited)")->group("Resources");
    double memoryBudgetGB = 0.0;
    app.add_option("--memory-budget", memoryBudgetGB,
        "Advisory unified-memory budget in GB; derives --max-splats if unset")
        ->group("Resources");

    // ── Progress / verbosity (M1) ──
    int progressEvery = 100;
    app.add_option("--progress-every", progressEvery,
        "Emit a progress line every N steps (0 = off)")->group("Progress");
    std::string progressFormat = "plain";
    app.add_option("--progress-format", progressFormat, "Progress line format: plain | jsonl")
        ->check(CLI::IsMember({"plain", "jsonl"}))->group("Progress");
    bool quiet = false, verbose = false;
    app.add_flag("--quiet", quiet, "Suppress progress and per-event logs (errors still shown)")->group("Progress");
    app.add_flag("--verbose", verbose, "Verbose logging")->group("Progress");

    CLI11_PARSE(app, argc, argv);

    // Apply preset to options the user left at default.
    if (!preset.empty()) {
        if (preset == "draft") {
            if (!optIters->count())     numIters = 7000;
            if (!optDownscale->count()) downScaleFactor = 2.0f;
            if (!shDegreeOpt->count())  shDegree = 2;
            if (!optMaxSplats->count()) maxSplats = 1000000;
        } else if (preset == "balanced") {
            if (!optIters->count())     numIters = 30000;
            if (!optMaxSplats->count()) maxSplats = 3000000;
        } else if (preset == "production") {
            if (!optIters->count())     numIters = 100000;
            if (!optMaxSplats->count()) maxSplats = 6000000;
        }
    }

    // Derive a splat cap from a memory budget if one wasn't set explicitly.
    // Heuristic: ~900 B/splat (params + 2× Adam + densify scratch + sort bins);
    // reserve ~45% of the budget for decoded images, framebuffers, and the OS.
    if (memoryBudgetGB > 0.0 && !optMaxSplats->count()) {
        maxSplats = msplat::maxSplatsFromBudgetGB(memoryBudgetGB);
        if (!quiet)
            std::cout << "memory-budget " << memoryBudgetGB << " GB -> --max-splats "
                      << maxSplats << " (advisory)" << std::endl;
    }

    int verbosity = quiet ? 0 : (verbose ? 2 : 1);
    if (debugBg && !optBgColor->count()) bgColor = {0.6130f, 0.0101f, 0.3984f};

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    int errPhase = EXIT_LOAD;   // bumped to a generic code once we're past loading
    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        for (auto &cam : inputData.cameras)
            cam.loadImage(downScaleFactor);

        // Over-downscale guard: tiny render resolutions destabilize training (the
        // optimizer diverges, not just lower quality — seen on small-image datasets
        // like Tanks&Temples at -d4 → 244px). Warn and suggest a smaller -d.
        if (!inputData.cameras.empty()) {
            int minLong = INT_MAX;
            for (auto &cam : inputData.cameras)
                minLong = std::min(minLong, std::max(cam.width, cam.height));
            if (msplat::isCoarseRender(minLong, 0))
                std::cerr << "warning: render resolution is only ~" << minLong << "px on the long side"
                          << " (after -d " << downScaleFactor << "). Training may be unstable at this size; "
                          << "try a smaller --downscale-factor (these images are already small)." << std::endl;
        }

        std::vector<Camera> cams;
        std::vector<Camera> testCams;
        Camera *valCam = nullptr;

        if (evalMode) {
            auto [train, test] = inputData.splitTrainTest(testEvery);
            cams = train; testCams = test;
            // Kept verbatim — gsplata's parser matches this exact line.
            std::cout << "Eval mode: " << cams.size() << " train, " << testCams.size() << " test" << std::endl;
        } else {
            auto [train, val] = inputData.getCameras(validate, valImage);
            cams = train; valCam = val;
        }

        Model model(inputData, cams.size(),
                     numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                     refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh,
                     densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                     numIters, keepCrs,
                     bgColor.data());
        model.maxSplats = (int)maxSplats;
        model.quiet = (verbosity == 0);
        errPhase = 1;   // dataset + model are up; later failures are generic (1)

        install_signal_handlers();

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;

        bool benchmarking = std::getenv("BENCHMARK") != nullptr;
        int bench_warmup = 50;
        std::vector<double> bench_iter_ms, bench_cpu_ms, bench_drain_ms;
        if (benchmarking) {
            bench_iter_ms.reserve(numIters);
            bench_cpu_ms.reserve(numIters);
            bench_drain_ms.reserve(numIters);
        }
        auto cpu_now = []() { return std::chrono::high_resolution_clock::now(); };

        // Progress reporting state (M1). Loss is read sync-free from the previous
        // iteration's buffer (msplat_train_step does not stall), so progress is
        // effectively free. Lines are flushed per write and never gated on a TTY.
        auto prog_anchor = cpu_now();
        size_t prog_anchor_step = step - 1;
        bool jsonl = (progressFormat == "jsonl");
        bool emitProgress = (verbosity >= 1 && progressEvery > 0);

        auto bench_start = cpu_now();
        size_t lastStep = step - 1;
        for (; step <= (size_t)numIters; step++) {
            if (g_caught_signal.load(std::memory_order_relaxed) != 0) { lastStep = step - 1; break; }
            lastStep = step;
            Camera &cam = cams[camsIter.next()];

            auto iter_start = cpu_now();
            MTensor gt = cam.getGPUImage(model.getDownscaleFactor(step));
            model.fullIteration(cam, step, gt, ssimWeight);
            model.schedulersStep(step);
            model.afterTrain(step);
            msplat_commit();

            if (emitProgress && (step % (size_t)progressEvery == 0 || step == (size_t)numIters)) {
                // Drain the pipeline so loss_sum holds this step's real loss. Amortized
                // cost ≈ one pipeline drain per progressEvery iters (<1% at default 100).
                msplat_gpu_sync();
                auto now = cpu_now();
                double span_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - prog_anchor).count() / 1000.0;
                size_t span_steps = step - prog_anchor_step;
                double ms_per = span_steps ? span_ms / (double)span_steps : 0.0;
                prog_anchor = now; prog_anchor_step = step;
                long splats = (long)model.means.size(0);
                float loss = msplat_read_loss(model.lastHeight, model.lastWidth);
                printf("%s\n", msplat::formatProgress(jsonl, step, numIters, splats, loss, ms_per).c_str());
                fflush(stdout);
            }

            if (benchmarking && step > (size_t)bench_warmup) {
                auto pre_sync = cpu_now();
                msplat_gpu_sync();
                auto iter_end = cpu_now();
                double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(pre_sync - iter_start).count() / 1000.0;
                double drain_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - pre_sync).count() / 1000.0;
                bench_iter_ms.push_back(iter_ms);
                bench_cpu_ms.push_back(cpu_ms);
                bench_drain_ms.push_back(drain_ms);
            }

            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            if (!valRender.empty() && step % 10 == 0) {
                MTensor rgb = model.render(*valCam, step);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                Image valImg;
                valImg.width = (int)rgb_cpu.size(1);
                valImg.height = (int)rgb_cpu.size(0);
                valImg.data.resize(valImg.width * valImg.height * 3);
                memcpy(valImg.ptr(), rgb_cpu.data_ptr(), valImg.data.size() * sizeof(float));
                imwriteRGB((fs::path(valRender) / (std::to_string(step) + ".png")).string(), valImg);
            }
        }

        if (benchmarking && !bench_iter_ms.empty()) {
            auto bench_end = cpu_now();
            double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start).count() / 1000.0;
            size_t n = bench_iter_ms.size();
            std::vector<double> sorted = bench_iter_ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            double mean = sum / n;
            double median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
            double sq_sum = 0;
            for (double v : sorted) sq_sum += (v - mean) * (v - mean);
            double stddev = std::sqrt(sq_sum / n);

            std::cout << "\n=== Benchmark (" << n << " iters, " << bench_warmup << " warmup, " << total_s << "s total) ===\n";
            std::cout << "  mean:   " << mean   << " ms/iter\n";
            std::cout << "  median: " << median  << " ms/iter\n";
            std::cout << "  stddev: " << stddev  << " ms/iter\n";
            std::cout << "  p5:     " << sorted[(size_t)(n * 0.05)] << " ms/iter\n";
            std::cout << "  p95:    " << sorted[(size_t)(n * 0.95)] << " ms/iter\n";
            std::cout << "  min:    " << sorted.front() << " ms/iter\n";
            std::cout << "  max:    " << sorted.back()  << " ms/iter\n";
            std::cout << "  wall:   " << total_s << "s for " << numIters << " iters\n";

            auto stats = [](std::vector<double> &v) {
                std::vector<double> s = v;
                std::sort(s.begin(), s.end());
                size_t n = s.size();
                double sum = std::accumulate(s.begin(), s.end(), 0.0);
                double med = (n % 2 == 0) ? (s[n/2-1] + s[n/2]) / 2.0 : s[n/2];
                return std::make_pair(sum / n, med);
            };
            auto [cpu_mean, cpu_med] = stats(bench_cpu_ms);
            auto [drain_mean, drain_med] = stats(bench_drain_ms);
            std::cout << "\n  --- CPU dispatch vs GPU drain ---\n";
            std::cout << "  cpu dispatch:  mean=" << cpu_mean << "  median=" << cpu_med << " ms\n";
            std::cout << "  gpu drain:     mean=" << drain_mean << "  median=" << drain_med << " ms\n";
            std::cout << "  gpu fraction:  " << (drain_med / median * 100) << "%\n";

            // GPU timing from completion handlers (PROFILE_GPU=1)
            std::vector<double> gpu_times;
            msplat_drain_gpu_times(gpu_times);
            if (!gpu_times.empty()) {
                auto [gpu_mean, gpu_med] = stats(gpu_times);
                std::vector<double> gs = gpu_times;
                std::sort(gs.begin(), gs.end());
                std::cout << "\n  --- GPU kernel time (from CB completion handlers) ---\n";
                std::cout << "  gpu exec:   mean=" << gpu_mean << "  median=" << gpu_med << " ms\n";
                std::cout << "  gpu p5:     " << gs[(size_t)(gs.size() * 0.05)] << " ms\n";
                std::cout << "  gpu p95:    " << gs[(size_t)(gs.size() * 0.95)] << " ms\n";
                std::cout << "  gpu min:    " << gs.front() << " ms\n";
                std::cout << "  gpu max:    " << gs.back() << " ms\n";
                std::cout << "  n_cbs:      " << gs.size() << "\n";
            }

            // Per-stage GPU timing (PROFILE_STAGES=1)
            constexpr int MAX_STAGES = 16;
            std::vector<double> stage_times[MAX_STAGES];
            const char* stage_names[MAX_STAGES] = {};
            int n_stages = 0;
            msplat_drain_stage_times(stage_times, MAX_STAGES, n_stages, stage_names);
            bool has_stage_data = false;
            for (int i = 0; i < n_stages; i++) if (!stage_times[i].empty()) { has_stage_data = true; break; }
            if (has_stage_data) {
                std::cout << "\n  --- Per-stage GPU time (Metal timestamp counters) ---\n";
                double total_med = 0;
                for (int i = 0; i < n_stages; i++) {
                    if (stage_times[i].empty()) continue;
                    auto [s_mean, s_med] = stats(stage_times[i]);
                    total_med += s_med;
                    std::cout << "  " << std::left << std::setw(22) << stage_names[i]
                              << "median=" << std::fixed << std::setprecision(3) << s_med
                              << "ms  mean=" << s_mean << "ms  (" << stage_times[i].size() << " samples)\n";
                }
                std::cout << "  " << std::left << std::setw(22) << "TOTAL (sum medians)"
                          << std::fixed << std::setprecision(3) << total_med << "ms\n";
            }
            std::cout << "\n";
        }

        // ── Interrupted by SIGINT/SIGTERM: write a partial scene and exit ──
        int caught = g_caught_signal.load(std::memory_order_relaxed);
        if (caught != 0) {
            fs::path p(outputScene);
            std::string interruptedPath =
                p.replace_filename(p.stem().string() + "_interrupted" + p.extension().string()).string();
            std::cerr << "\nReceived " << (caught == SIGINT ? "SIGINT" : "SIGTERM")
                      << " at step " << lastStep << "; saving partial scene…" << std::endl;
            try { model.save(interruptedPath, (int)lastStep); } catch (...) {}
            std::cout << "Done: " << lastStep << " iters (interrupted), "
                      << model.means.size(0) << " Gaussians, wrote "
                      << fs::absolute(interruptedPath).string() << std::endl;
            fflush(stdout);
            cleanup_msplat_metal();
            msplat_gpu_sync();
            return 128 + caught;   // 130 (SIGINT) / 143 (SIGTERM)
        }

        // ── Normal completion: write cameras.json + final scene ──
        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, numIters);
        if (!fs::exists(outputScene) || fs::file_size(outputScene) == 0) {
            std::cerr << "error: failed to write output scene: " << outputScene << std::endl;
            cleanup_msplat_metal(); msplat_gpu_sync();
            return EXIT_WRITE;
        }

        double meanPsnr = 0.0;
        bool havePsnr = false;

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], numIters);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = testCams[i].getGPUImage(model.getDownscaleFactor(numIters)).cpu();

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            meanPsnr = sumPsnr / nTest; havePsnr = true;
            std::cout << "\n  PSNR:  " << meanPsnr
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, numIters);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters)).cpu();

            float p = psnr(rgb_cpu, gt_cpu);
            if (!havePsnr) { meanPsnr = p; havePsnr = true; }
            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << p
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // ── Final machine-readable summary line (M1.3) ──
        // Stable, single line; gsplata harvests the absolute PLY path from here.
        std::cout << "Done: " << numIters << " iters, " << model.means.size(0) << " Gaussians";
        if (havePsnr) std::cout << ", PSNR " << std::fixed << std::setprecision(2) << meanPsnr;
        std::cout << ", wrote " << fs::absolute(outputScene).string() << std::endl;
        fflush(stdout);

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        // errPhase is EXIT_LOAD (3) until the dataset+model are built, then 1.
        return errPhase;
    }
    return EXIT_OK;
}
