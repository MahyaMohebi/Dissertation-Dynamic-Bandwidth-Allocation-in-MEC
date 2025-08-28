#include "../../include/abr/abr_mpc.h"
#include <algorithm>

namespace vqm {

AbrDecision AbrMpc::Decide(const AbrContext& ctx) {
    AbrDecision out{};
    out.policy = Name();
    if (ctx.throughput_mbps > 0) {
        m_tp_hist.push_back(ctx.throughput_mbps);
        while (m_tp_hist.size() > m_hist) m_tp_hist.pop_front();
    }
    double avg = ctx.throughput_mbps;
    if (!m_tp_hist.empty()) {
        double s=0; for (double v: m_tp_hist) s+=v; avg = s / m_tp_hist.size();
    }
    double budget_mbps = 0.8 * avg;
    int chosen = ctx.ladder_kbps.empty() ? ctx.last_bitrate_kbps : ctx.ladder_kbps.front();
    for (int br : ctx.ladder_kbps) {
        double br_mbps = br / 1000.0;
        if (br_mbps <= budget_mbps) chosen = br;
    }
    out.bitrate_kbps = chosen;
    out.reason = "avg_tp_budget";
    return out;
}

} // namespace vqm
