#include "../../include/cache/cache_policies.h"
#include <cassert>

namespace vqm {

void LruCachePolicy::Put(const CacheEntry& e) {
    if (e.size_bytes > m_capacity) return; // too big
    if (m_map.find(e.key) != m_map.end()) {
        Touch(e.key);
        return;
    }
    while (m_used + e.size_bytes > m_capacity) {
        if (m_lru.empty()) break;
        auto back = m_lru.back();
        auto it = m_map.find(back);
        if (it != m_map.end()) {
            m_used -= it->second.first;
            m_lru.pop_back();
            m_map.erase(it);
        } else {
            m_lru.pop_back();
        }
    }
    m_lru.push_front(e.key);
    m_map[e.key] = {e.size_bytes, m_lru.begin()};
    m_used += e.size_bytes;
}

bool LruCachePolicy::Contains(const SegmentKey& k) const {
    return m_map.find(k) != m_map.end();
}

void LruCachePolicy::Touch(const SegmentKey& k) {
    auto it = m_map.find(k);
    if (it == m_map.end()) return;
    m_lru.erase(it->second.second);
    m_lru.push_front(k);
    it->second.second = m_lru.begin();
}

void LruCachePolicy::EvictUntil(size_t target_free_bytes) {
    size_t need_free = (m_used > target_free_bytes) ? (m_used - target_free_bytes) : 0;
    while (need_free > 0 && !m_lru.empty()) {
        auto back = m_lru.back();
        auto it = m_map.find(back);
        if (it == m_map.end()) { m_lru.pop_back(); continue; }
        size_t sz = it->second.first;
        if (sz > need_free) need_free = 0; else need_free -= sz;
        m_used -= sz;
        m_lru.pop_back();
        m_map.erase(it);
    }
}

void LruCachePolicy::Clear() {
    m_lru.clear();
    m_map.clear();
    m_used = 0;
}

} // namespace vqm
