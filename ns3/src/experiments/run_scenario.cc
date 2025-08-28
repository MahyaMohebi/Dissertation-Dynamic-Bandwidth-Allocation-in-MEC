#include "../../include/experiments/scenario.h"
#include "../../include/abr/abr_bola.h"
#include "../../include/abr/abr_mpc.h"
#include "../../include/abr/abr_pensive.h"
#include "../../include/predict/stub_predictor.h"
#include "../../include/utils/qoe_logger.h"
#include "../../include/apps/dash_client.h"
#include <memory>
#include <iostream>
#include <string>
#include <vector>

namespace vqm {

static std::shared_ptr<AbrController> MakeAbr(AbrKind k) {
    switch (k) {
        case AbrKind::kBola:   return std::make_shared<AbrBola>();
        case AbrKind::kMpc:    return std::make_shared<AbrMpc>();
        case AbrKind::kPensive:return std::make_shared<AbrPensive>();
    }
    return std::make_shared<AbrBola>();
}

static AbrKind ParseAbr(const std::string& s) {
    if (s=="BOLA") return AbrKind::kBola;
    if (s=="MPC") return AbrKind::kMpc;
    if (s=="Pensive") return AbrKind::kPensive;
    return AbrKind::kBola;
}

int BuildAndRun(const ScenarioConfig& cfg) {
    auto abr = MakeAbr(cfg.abr);
    PredictorPtr predictor = std::make_shared<StubPredictor>(5);
    auto logger = std::make_shared<QoeLogger>(cfg.results_dir + "/" + cfg.run_id + "_decisions.csv");
    DashClient client(abr, predictor, logger);
    VideoProfile vp;
    vp.ladder_kbps = cfg.ladder_kbps;
    vp.segment_duration_s = cfg.segment_duration_s;
    vp.buffer_target_s = cfg.buffer_target_s;
    client.Configure(vp);
    client.SetClientId("client0");
    client.EnableCache(cfg.enable_cache);

    for (int i=0; i<5; ++i) client.RequestNextSegment();

    logger->Flush();
    std::cout << "Stub run complete: wrote " << (cfg.results_dir + "/" + cfg.run_id + "_decisions.csv") << "\n";
    return 0;
}

} // namespace vqm

// Simple CLI to set ScenarioConfig fields
int main(int argc, char** argv) {
    vqm::ScenarioConfig cfg;
    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string& flag)->std::string{
            if (i+1 < argc) return argv[++i];
            return "";
        };
        if (a=="--results_dir") cfg.results_dir = next(a);
        else if (a=="--run_id") cfg.run_id = next(a);
        else if (a=="--abr") cfg.abr = vqm::ParseAbr(next(a));
        else if (a=="--cache_policy") cfg.cache_policy = next(a);
        else if (a=="--enable_cache") cfg.enable_cache = (next(a)!="0");
        else if (a=="--enable_prediction") cfg.enable_prediction = (next(a)!="0");
        else if (a=="--segment_duration_s") cfg.segment_duration_s = std::stod(next(a));
        else if (a=="--buffer_target_s") cfg.buffer_target_s = std::stod(next(a));
        else if (a=="--ladder_kbps") {
            std::string s = next(a);
            cfg.ladder_kbps.clear();
            std::string num;
            for (char c : s) {
                if (c==',' || c==' ') { if (!num.empty()) { cfg.ladder_kbps.push_back(std::stoi(num)); num.clear(); } }
                else num.push_back(c);
            }
            if (!num.empty()) cfg.ladder_kbps.push_back(std::stoi(num));
        }
        else if (a=="--trace_csv") cfg.trace_csv = next(a);
    }
    return vqm::BuildAndRun(cfg);
}
