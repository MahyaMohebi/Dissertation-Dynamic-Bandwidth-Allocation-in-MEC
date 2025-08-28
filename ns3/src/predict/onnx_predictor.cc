#include "../../include/predict/onnx_predictor.h"
#include <memory>

namespace vqm {

struct OnnxPredictor::Impl {
    bool ready = false;
    std::string path;
};

OnnxPredictor::OnnxPredictor() : m_impl(new Impl()) {}
OnnxPredictor::~OnnxPredictor() = default;

bool OnnxPredictor::Load(const std::string& artifact_path) {
    m_impl->path = artifact_path;
    // Stub: In real build, initialize ONNX Runtime session here.
    m_impl->ready = false; // not actually loading yet
    return false;
}
bool OnnxPredictor::IsReady() const { return m_impl->ready; }

Prediction OnnxPredictor::Predict(const std::vector<double>&) {
    // Not ready; return invalid
    return Prediction{0.0, 1.0, false};
}

void OnnxPredictor::Warmup() {}

} // namespace vqm
