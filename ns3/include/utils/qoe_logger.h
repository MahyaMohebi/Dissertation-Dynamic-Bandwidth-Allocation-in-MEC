#ifndef VIDEO_QOE_MEC_QOE_LOGGER_H
#define VIDEO_QOE_MEC_QOE_LOGGER_H
#include <string>
#include <vector>
#include <fstream>
#include <memory>

namespace vqm {

struct DecisionLog {
    double t_s;
    int seg_index;
    int bitrate_kbps;
    std::string policy;
    std::string reason;
};

struct RebufferLog {
    double t_s;
    double duration_s;
};

struct DownloadLog {
    double t_s;
    int seg_index;
    int bitrate_kbps;
    double size_bytes;
    double dl_time_s;
    bool cache_hit;
    std::string source;
};

class QoeLogger {
public:
    explicit QoeLogger(const std::string& out_csv_path);
    ~QoeLogger();

    void LogDecision(const DecisionLog& d);
    void LogRebuffer(const RebufferLog& r);
    void LogDownload(const DownloadLog& d);

    void Flush();
private:
    std::string m_path;
    std::ofstream m_of;
};

} // namespace vqm
#endif
