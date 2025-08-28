#include "../../include/predict/stub_predictor.h"
#include <algorithm>

namespace vqm {

Prediction StubPredictor::Predict(const std::vector<double>& features) {
    Prediction p{};
    if (features.empty()) { p.valid = false; p.mbps = 0.0; return p; }
    double s=0; size_t n = std::min(m_win, features.size());
    for (size_t i=features.size()-n; i<features.size(); ++i) s += features[i];
    p.mbps = s / n;
    p.horizon_s = 1.0;
    p.valid = true;
    return p;
}

} // namespace vqm
