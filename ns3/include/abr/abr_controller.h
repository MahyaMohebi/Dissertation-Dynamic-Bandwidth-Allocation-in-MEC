#ifndef VIDEO_QOE_MEC_ABR_CONTROLLER_H
#define VIDEO_QOE_MEC_ABR_CONTROLLER_H
/**
 * Adaptive Bitrate (ABR) Controller - abstract interface.
 * Goal: reduce rebuffering while maintaining quality and smoothness.
 *
 * Implementations: BOLA, MPC, Pensive, plus your hybrid (later).
 */
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace vqm {

struct AbrContext {
    double now_s {0.0};          // simulation time [s]
    double throughput_mbps {0.0}; // last measured net throughput [Mbps]
    double buffer_s {0.0};        // current buffered video [s]
    int last_bitrate_kbps {0};    // previously selected representation bitrate
    int segment_index {0};        // 0-based next segment
    std::vector<int> ladder_kbps; // available bitrates
};

struct AbrDecision {
    int bitrate_kbps {0};
    std::string policy;        // e.g., "BOLA"
    std::string reason;        // short tag (e.g., "throughput", "buffer", "predicted")
};

class AbrController {
public:
    virtual ~AbrController() = default;
    virtual std::string Name() const = 0;
    virtual AbrDecision Decide(const AbrContext& ctx) = 0;
};

} // namespace vqm
#endif
