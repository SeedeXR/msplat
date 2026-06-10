#ifndef MSPLAT_PLY_HEADER_HPP
#define MSPLAT_PLY_HEADER_HPP

// Pure (no-GPU) construction and parsing of the 3DGS PLY header. Both the writer
// (save_gaussians.cpp) and reader use the SAME step-comment prefix from here, so a
// reader/writer mismatch like the historical "--resume lost its step" bug cannot
// recur silently — it is guarded by a unit test in tests/cpp.

#include <string>
#include <istream>

namespace msplat {

// The resume step is encoded as a PLY comment: "comment msplat v<step>".
inline const std::string& plyStepCommentPrefix() {
    static const std::string p = "comment msplat v";
    return p;
}

struct PlyHeaderInfo {
    long numPoints = 0;   // element vertex count
    int  numDc = 0;       // count of f_dc_* properties (3 for RGB DC)
    int  numFr = 0;       // count of f_rest_* properties (3 * (bases-1))
    int  step = 0;        // training iteration recorded by the writer (0 if absent)
    bool isPly = false;   // first line was "ply"
};

// Parse a 3DGS PLY header from a stream positioned at byte 0, consuming up to and
// including the "end_header" line (so the stream is left at the binary payload).
inline PlyHeaderInfo parsePlyHeader(std::istream& f) {
    PlyHeaderInfo info;
    std::string line;
    if (!std::getline(f, line)) return info;
    info.isPly = (line.rfind("ply", 0) == 0);  // must START with "ply" (CR-tolerant)

    const std::string& stepPrefix = plyStepCommentPrefix();
    const std::string vertexPrefix = "element vertex ";
    while (std::getline(f, line)) {
        if (line == "end_header" || line == "end_header\r") break;
        if (line.rfind(stepPrefix, 0) == 0) {
            try { info.step = std::stoi(line.substr(stepPrefix.size())); } catch (...) { info.step = 0; }
        } else if (line.rfind(vertexPrefix, 0) == 0) {
            try { info.numPoints = std::stol(line.substr(vertexPrefix.size())); } catch (...) {}
        } else if (line.rfind("property float f_dc_", 0) == 0) {
            info.numDc++;
        } else if (line.rfind("property float f_rest_", 0) == 0) {
            info.numFr++;
        }
    }
    return info;
}

}  // namespace msplat

#endif
