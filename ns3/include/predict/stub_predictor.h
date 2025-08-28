#ifndef VIDEO_QOE_MEC_STUB_PREDICTOR_H
#define VIDEO_QOE_MEC_STUB_PREDICTOR_H
#include "predictor_iface.h"
#include <deque>

namespace vqm {

class StubPredictor final : public Predictor {
public:
    explicit StubPredictor(size_t window=5) : m_win(window) {}
    std::string Name() const override { return "stub_ma"; }
    bool Load(const std::string&) override { return true; }
    bool IsReady() const override { return true; }
    Prediction Predict(const std::vector<double>& features) override;
private:
    size_t m_win;
    std::deque<double> m_hist;
};

} // namespace vqm
#endif
