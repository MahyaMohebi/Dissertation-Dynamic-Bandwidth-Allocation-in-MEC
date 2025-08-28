#include "../../include/utils/qoe_logger.h"
#include <iomanip>

namespace vqm {

QoeLogger::QoeLogger(const std::string& out_csv_path) : m_path(out_csv_path), m_of(out_csv_path) {
    if (m_of.is_open()) {
        m_of << "type,t_s,seg_index,bitrate_kbps,policy,reason,duration_s,size_bytes,dl_time_s,cache_hit,source\n";
    }
}
QoeLogger::~QoeLogger() { Flush(); }

void QoeLogger::LogDecision(const DecisionLog& d) {
    if (!m_of.is_open()) return;
    m_of << "decision," << d.t_s << "," << d.seg_index << "," << d.bitrate_kbps
         << "," << d.policy << "," << d.reason << ",,,,," << "\n";
}
void QoeLogger::LogRebuffer(const RebufferLog& r) {
    if (!m_of.is_open()) return;
    m_of << "rebuffer," << r.t_s << ",,,"
         << ",," << r.duration_s << ",,,," << "\n";
}
void QoeLogger::LogDownload(const DownloadLog& d) {
    if (!m_of.is_open()) return;
    m_of << "download," << d.t_s << "," << d.seg_index << "," << d.bitrate_kbps
         << ",,," << "," << "," << d.size_bytes << "," << d.dl_time_s
         << "," << (d.cache_hit ? 1 : 0) << "," << d.source << "\n";
}
void QoeLogger::Flush() {
    if (m_of.is_open()) m_of.flush();
}

} // namespace vqm
