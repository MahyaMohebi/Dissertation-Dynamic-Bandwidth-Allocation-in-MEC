#include "../../include/abr/abr_bola.h"
#include <algorithm>

namespace vqm {

AbrDecision AbrBola::Decide(const AbrContext& ctx) {
    AbrDecision out{};
    out.policy = Name();
    if (ctx.ladder_kbps.empty()) { out.bitrate_kbps = ctx.last_bitrate_kbps; out.reason = "no_ladder"; return out; }
    int minBR = ctx.ladder_kbps.front();
    int maxBR = ctx.ladder_kbps.back();
    double b = ctx.buffer_s;
    if (b <= m_reservoir) {
        out.bitrate_kbps = minBR; out.reason = "low_buffer";
    } else if (b >= (m_reservoir + m_cushion)) {
        out.bitrate_kbps = maxBR; out.reason = "high_buffer";
    } else {
        double frac = (b - m_reservoir) / std::max(1e-6, m_cushion);
        size_t idx = static_cast<size_t>(frac * (ctx.ladder_kbps.size()-1));
        if (idx >= ctx.ladder_kbps.size()) idx = ctx.ladder_kbps.size()-1;
        out.bitrate_kbps = ctx.ladder_kbps[idx];
        out.reason = "buffer_interpolate";
    }
    return out;
}

} // namespace vqm
