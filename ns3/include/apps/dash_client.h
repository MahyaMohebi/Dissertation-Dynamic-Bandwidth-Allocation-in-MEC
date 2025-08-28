#ifndef VIDEO_QOE_MEC_DASH_CLIENT_H
#define VIDEO_QOE_MEC_DASH_CLIENT_H
/**
 * DASH client app (ns-3 Application to be completed later).
 * Cooperates with ABR and (optional) Predictor to request segments,
 * logs QoE signals, and interfaces with Edge/Origin servers.
 */
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include "../utils/qoe_logger.h"
#include "../predict/predictor_iface.h"
#include "../abr/abr_controller.h"

namespace ns3 { class Application; }

namespace vqm {

struct VideoProfile {
    std::vector<int> ladder_kbps; // e.g., [300, 750, 1200, 2500, 4000]
    double segment_duration_s {2.0};
    double buffer_target_s {20.0};
};

class DashClient /* : public ns3::Application */ {
public:
    DashClient(std::shared_ptr<AbrController> abr,
               PredictorPtr predictor,
               std::shared_ptr<QoeLogger> logger);

    void Configure(const VideoProfile& profile);
    void SetClientId(std::string id) { m_clientId = std::move(id); }
    void SetServers(const std::string& edgeAddr, const std::string& originAddr);
    void EnableCache(bool on) { m_useCache = on; }

    // Called by the scenario to simulate the pulling of one segment.
    void RequestNextSegment();

private:
    std::string m_clientId {"client0"};
    std::shared_ptr<AbrController> m_abr;
    PredictorPtr m_predictor;
    std::shared_ptr<QoeLogger> m_logger;
    VideoProfile m_profile;
    bool m_useCache {true};
    int m_nextSeg {0};
    int m_lastBitrate {0};
    double m_buffer_s {0.0};
};

} // namespace vqm
#endif
