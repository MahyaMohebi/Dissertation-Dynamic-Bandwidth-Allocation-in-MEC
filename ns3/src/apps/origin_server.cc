#include "../../include/apps/origin_server.h"

namespace vqm {

size_t OriginServer::SegmentSizeBytes(int bitrate_kbps, double segment_s) const {
    double bytes = (static_cast<double>(bitrate_kbps) * 1000.0 / 8.0) * segment_s;
    return static_cast<size_t>(bytes);
}

} // namespace vqm
