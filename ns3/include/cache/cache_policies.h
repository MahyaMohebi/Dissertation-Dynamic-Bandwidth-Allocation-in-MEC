#ifndef VIDEO_QOE_MEC_CACHE_POLICIES_H
#define VIDEO_QOE_MEC_CACHE_POLICIES_H
#include <unordered_map>
#include <list>
#include <string>
#include <memory>
#include <optional>

namespace vqm {

using SegmentKey = std::string; // e.g., "video:seg_001_3000kbps"

struct CacheEntry {
    SegmentKey key;
    size_t size_bytes {0};
};

class CachePolicy {
public:
    virtual ~CachePolicy() = default;
    virtual std::string Name() const = 0;
    virtual void Put(const CacheEntry& e) = 0;
    virtual bool Contains(const SegmentKey& k) const = 0;
    virtual void Touch(const SegmentKey& k) = 0;
    virtual void EvictUntil(size_t target_free_bytes) = 0;
    virtual void Clear() = 0;
};

class NoCachePolicy final : public CachePolicy {
public:
    std::string Name() const override { return "nocache"; }
    void Put(const CacheEntry&) override {}
    bool Contains(const SegmentKey&) const override { return false; }
    void Touch(const SegmentKey&) override {}
    void EvictUntil(size_t) override {}
    void Clear() override {}
};

class LruCachePolicy final : public CachePolicy {
public:
    explicit LruCachePolicy(size_t capacity_bytes) : m_capacity(capacity_bytes) {}
    std::string Name() const override { return "lru"; }
    void Put(const CacheEntry& e) override;
    bool Contains(const SegmentKey& k) const override;
    void Touch(const SegmentKey& k) override;
    void EvictUntil(size_t target_free_bytes) override;
    void Clear() override;
private:
    size_t m_capacity;
    size_t m_used {0};
    std::list<SegmentKey> m_lru;
    std::unordered_map<SegmentKey, std::pair<size_t, std::list<SegmentKey>::iterator>> m_map;
};

} // namespace vqm
#endif
