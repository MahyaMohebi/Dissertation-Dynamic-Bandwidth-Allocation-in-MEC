#ifndef VIDEO_QOE_MEC_ONNX_PREDICTOR_H
#define VIDEO_QOE_MEC_ONNX_PREDICTOR_H
#include "predictor_iface.h"
#include <memory>
#include <string>

namespace vqm {

// Forward declare ONNX Runtime types to avoid heavy includes in the header.
namespace ort { class Env; class Session; }

class OnnxPredictor final : public Predictor {
public:
    OnnxPredictor();
    ~OnnxPredictor() override;
    std::string Name() const override { return "onnx_lstm"; }
    bool Load(const std::string& artifact_path) override;
    bool IsReady() const override;
    Prediction Predict(const std::vector<double>& features) override;
    void Warmup() override;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl; // PIMPL to keep header light
};

} // namespace vqm
#endif
