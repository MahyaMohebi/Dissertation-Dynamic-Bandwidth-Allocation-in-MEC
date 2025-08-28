#ifndef VIDEO_QOE_MEC_SCENARIO_H
#define VIDEO_QOE_MEC_SCENARIO_H
#include <string>
#include <memory>
#include <vector>

namespace vqm {

enum class AbrKind { kBola, kMpc, kPensive };

struct ScenarioConfig {
    std::string trace_csv;       // relative to datasets/
    AbrKind abr {AbrKind::kBola};
    bool enable_prediction {true};
    bool enable_cache {true};
    std::string cache_policy {"lru"}; // or "nocache"
    size_t cache_capacity_bytes {64ull * 1024 * 1024}; // 64 MiB default
    double segment_duration_s {2.0};
    double buffer_target_s {20.0};
    std::vector<int> ladder_kbps {300, 750, 1200, 2500, 4000};
    std::string results_dir {"results/logs"};
    std::string run_id {"run_000"};
};

// To be implemented in src/experiments/run_scenario.cc
int BuildAndRun(const ScenarioConfig& cfg);

} // namespace vqm
#endif
