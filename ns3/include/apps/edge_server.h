#ifndef VIDEO_QOE_MEC_EDGE_SERVER_H
#define VIDEO_QOE_MEC_EDGE_SERVER_H
#include <memory>
#include <string>
#include "../cache/edge_cache_model.h"

namespace vqm {

class EdgeServer {
public:
    EdgeServer(std::shared_ptr<EdgeCacheModel> cache, std::string name="edge0")
    : m_cache(std::move(cache)), m_name(std::move(name)) {}

    const std::string& Name() const { return m_name; }

    // Ask cache for a segment; return additional latency.
    CacheLookupResult Fetch(const SegmentKey& key, size_t size_bytes) {
        return m_cache->Fetch(key, size_bytes);
    }

private:
    std::shared_ptr<EdgeCacheModel> m_cache;
    std::string m_name;
};

} // namespace vqm
#endif
