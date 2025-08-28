#include "../../include/apps/dash_client.h"
#include <algorithm>

namespace vqm {

DashClient::DashClient(std::shared_ptr<AbrController> abr,
                       PredictorPtr predictor,
                       std::shared_ptr<QoeLogger> logger)
: m_abr(std::move(abr)), m_predictor(std::move(predictor)), m_logger(std::move(logger)) {}

void DashClient::Configure(const VideoProfile& profile) { m_profile = profile; }
void DashClient::SetServers(const std::string&, const std::string&) {}

void DashClient::RequestNextSegment() {
    AbrContext ctx;
    ctx.now_s = m_nextSeg * m_profile.segment_duration_s;
    ctx.buffer_s = std::max(0.0, m_profile.buffer_target_s - (m_nextSeg * m_profile.segment_duration_s));
    ctx.ladder_kbps = m_profile.ladder_kbps;
    ctx.last_bitrate_kbps = m_lastBitrate;
    double pred_mbps = 0.0;
    if (m_predictor && m_predictor->IsReady()) {
        Prediction p = m_predictor->Predict({});
        if (p.valid) pred_mbps = p.mbps;
    }
    ctx.throughput_mbps = pred_mbps;
    AbrDecision d = m_abr->Decide(ctx);
    m_lastBitrate = d.bitrate_kbps;
    if (m_logger) {
        m_logger->LogDecision(DecisionLog{
            ctx.now_s, m_nextSeg, d.bitrate_kbps, d.policy, d.reason
        });
    }
    ++m_nextSeg;
}

} // namespace vqm
