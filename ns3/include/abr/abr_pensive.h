#ifndef VIDEO_QOE_MEC_ABR_PENSIVE_H
#define VIDEO_QOE_MEC_ABR_PENSIVE_H
#include "abr_controller.h"
#include <string>

namespace vqm {

class AbrPensive final : public AbrController {
public:
    std::string Name() const override { return "Pensive"; }
    AbrDecision Decide(const AbrContext& ctx) override;
};

} // namespace vqm
#endif
