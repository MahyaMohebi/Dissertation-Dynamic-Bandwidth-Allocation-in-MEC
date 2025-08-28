#ifndef VIDEO_QOE_MEC_ORIGIN_SERVER_H
#define VIDEO_QOE_MEC_ORIGIN_SERVER_H
#include <string>
#include <cstddef>

namespace vqm {

class OriginServer {
public:
    explicit OriginServer(std::string name="origin0") : m_name(std::move(name)) {}
    const std::string& Name() const { return m_name; }
    // Placeholder: latency could depend on network; here we expose a base value.
    double BaseLatencyMs() const { return 25.0; }
    size_t SegmentSizeBytes(int bitrate_kbps, double segment_s) const;
private:
    std::string m_name;
};

} // namespace vqm
#endif
