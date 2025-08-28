#ifndef VIDEO_QOE_MEC_PREDICTOR_IFACE_H
#define VIDEO_QOE_MEC_PREDICTOR_IFACE_H
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace vqm {

struct Prediction {
    double mbps {0.0};    // predicted near-future throughput [Mbps]
    double horizon_s {1}; // prediction horizon [s]
    bool valid {false};
};

class Predictor {
public:
    virtual ~Predictor() = default;
    virtual std::string Name() const = 0;
    virtual bool Load(const std::string& artifact_path) = 0; // e.g., .onnx
    virtual bool IsReady() const = 0;
    virtual Prediction Predict(const std::vector<double>& features) = 0;
    virtual void Warmup() {}
};

using PredictorPtr = std::shared_ptr<Predictor>;

} // namespace vqm
#endif
