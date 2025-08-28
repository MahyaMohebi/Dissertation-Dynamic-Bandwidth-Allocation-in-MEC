#ifndef VIDEO_QOE_MEC_ABR_MPC_H
#define VIDEO_QOE_MEC_ABR_MPC_H
#include "abr_controller.h"
#include <deque>

namespace vqm {

class AbrMpc final : public AbrController {
public:
    explicit AbrMpc(size_t history = 5) : m_hist(history) {}
    std::string Name() const override { return "MPC"; }
    AbrDecision Decide(const AbrContext& ctx) override;
private:
    size_t m_hist;
    std::deque<double> m_tp_hist; // Mbps history
};

} // namespace vqm
#endif
