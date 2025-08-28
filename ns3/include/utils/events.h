#ifndef VIDEO_QOE_MEC_EVENTS_H
#define VIDEO_QOE_MEC_EVENTS_H
#include <functional>
#include <string>

namespace vqm {

using EventCallback = std::function<void()>;

enum class EventType {
    kRequestSegment,
    kDownloadComplete,
    kRebufferStart,
    kRebufferEnd
};

// Placeholder types for when we bind into ns-3's real scheduler later.
struct Event {
    double at_s {0.0};
    EventType type {EventType::kRequestSegment};
    EventCallback fn;
};

} // namespace vqm
#endif
