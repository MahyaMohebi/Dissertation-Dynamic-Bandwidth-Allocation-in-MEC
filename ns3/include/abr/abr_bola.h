#ifndef VIDEO_QOE_MEC_ABR_BOLA_H
#define VIDEO_QOE_MEC_ABR_BOLA_H
#include "abr_controller.h"
#include <string>

namespace vqm {

class AbrBola final : public AbrController {
public:
    explicit AbrBola(double reservoir_s = 5.0, double cushion_s = 10.0)
    : m_reservoir(reservoir_s), m_cushion(cushion_s) {}

    std::string Name() const override { return "BOLA"; }
    AbrDecision Decide(const AbrContext& ctx) override;

private:
    double m_reservoir;
    double m_cushion;
};

} // namespace vqm
#endif
