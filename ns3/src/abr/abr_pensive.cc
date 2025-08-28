#include "../../include/abr/abr_pensive.h"
#include <algorithm>

namespace vqm {

AbrDecision AbrPensive::Decide(const AbrContext& ctx) {
    AbrDecision out{};
    out.policy = Name();
    double budget_mbps = 0.7 * std::max(0.0, ctx.throughput_mbps);
    int chosen = ctx.ladder_kbps.empty() ? ctx.last_bitrate_kbps : ctx.ladder_kbps.front();
    for (int br : ctx.ladder_kbps) {
        if (br/1000.0 <= budget_mbps) chosen = br;
    }
    out.bitrate_kbps = chosen;
    out.reason = "throughput_factor";
    return out;
}

} // namespace vqm
