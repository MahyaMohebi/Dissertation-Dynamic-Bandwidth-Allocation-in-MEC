#ifndef VIDEO_QOE_MEC_EDGE_CACHE_MODEL_H
#define VIDEO_QOE_MEC_EDGE_CACHE_MODEL_H
#include "cache_policies.h"
#include <memory>
#include <string>
#include <cstddef>

namespace vqm {

struct CacheLookupResult {
    bool hit {false};
    double latency_ms {0.0}; // additional latency due to fetch path
    std::string source;      // "edge", "origin"
};

class EdgeCacheModel {
public:
    EdgeCacheModel(std::shared_ptr<CachePolicy> policy,
                   double edge_hit_ms = 5.0,
                   double origin_fetch_ms = 25.0)
    : m_policy(std::move(policy)), m_edge_hit_ms(edge_hit_ms), m_origin_fetch_ms(origin_fetch_ms) {}

    const CachePolicy& Policy() const { return *m_policy; }
    CachePolicy& Policy() { return *m_policy; }

    CacheLookupResult Fetch(const SegmentKey& key, size_t size_bytes);
    void Clear();
private:
    std::shared_ptr<CachePolicy> m_policy;
    double m_edge_hit_ms;
    double m_origin_fetch_ms;
};

} // namespace vqm
#endif
