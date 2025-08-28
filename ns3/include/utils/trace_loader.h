#ifndef VIDEO_QOE_MEC_TRACE_LOADER_H
#define VIDEO_QOE_MEC_TRACE_LOADER_H
#include <string>
#include <vector>
#include <utility>

namespace vqm {

struct ThroughputSample {
    double t_s {0.0};
    double mbps {0.0};
};

class TraceLoader {
public:
    // Loads a CSV with either 'time,mbps' or a single column of Mbps at 1s steps.
    static std::vector<ThroughputSample> LoadCsv(const std::string& path);
    // Convenience for 1 Hz samples: return Mbps values only.
    static std::vector<double> LoadMbps1Hz(const std::string& path);
};

} // namespace vqm
#endif
