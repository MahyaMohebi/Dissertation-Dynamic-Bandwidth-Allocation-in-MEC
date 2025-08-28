#include "../../include/utils/trace_loader.h"
#include <fstream>
#include <sstream>

namespace vqm {

std::vector<ThroughputSample> TraceLoader::LoadCsv(const std::string& path) {
    std::vector<ThroughputSample> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    bool header_checked=false;
    while (std::getline(f, line)) {
        if (!header_checked) { header_checked=true; /* naive: don't skip */ }
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string a,b;
        if (std::getline(ss, a, ',') && std::getline(ss, b, ',')) {
            try {
                double t = std::stod(a);
                double v = std::stod(b);
                out.push_back({t, v});
            } catch (...) { /* ignore parse errors */ }
        } else {
            try {
                double v = std::stod(line);
                double t = out.empty() ? 0.0 : out.back().t_s + 1.0;
                out.push_back({t, v});
            } catch (...) { /* ignore parse errors */ }
        }
    }
    return out;
}

std::vector<double> TraceLoader::LoadMbps1Hz(const std::string& path) {
    auto s = LoadCsv(path);
    std::vector<double> v; v.reserve(s.size());
    for (auto& e : s) v.push_back(e.mbps);
    return v;
}

} // namespace vqm
