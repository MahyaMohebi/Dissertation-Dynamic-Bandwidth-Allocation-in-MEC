#include "../../include/cache/edge_cache_model.h"

namespace vqm {

CacheLookupResult EdgeCacheModel::Fetch(const SegmentKey& key, size_t size_bytes) {
    CacheLookupResult r;
    if (m_policy->Contains(key)) {
        m_policy->Touch(key);
        r.hit = true;
        r.latency_ms = m_edge_hit_ms;
        r.source = "edge";
    } else {
        r.hit = false;
        r.latency_ms = m_origin_fetch_ms;
        r.source = "origin";
        m_policy->Put(CacheEntry{key, size_bytes});
    }
    return r;
}

void EdgeCacheModel::Clear() {
    m_policy->Clear();
}

} // namespace vqm
