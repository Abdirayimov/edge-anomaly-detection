#include "eanom/overlay/visualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <string>

namespace eanom::overlay {

namespace {

cv::Scalar severity_color(events::Severity s) {
    switch (s) {
        case events::Severity::Info:     return cv::Scalar(255, 200, 100);  // amber
        case events::Severity::Warning:  return cv::Scalar(40, 165, 240);   // orange
        case events::Severity::Critical: return cv::Scalar(40, 40, 240);    // red
    }
    return cv::Scalar(180, 180, 180);
}

void draw_heatmap(cv::Mat& frame, const cv::Mat& heatmap) {
    if (heatmap.empty()) return;
    cv::Mat h = heatmap;
    if (h.size() != frame.size()) {
        cv::resize(heatmap, h, frame.size(), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat hh = h.clone();
    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(hh, &min_v, &max_v);
    if (max_v <= 1e-6) return;
    cv::Mat normed;
    hh.convertTo(normed, CV_8U, 255.0 / max_v);
    cv::Mat colored;
    cv::applyColorMap(normed, colored, cv::COLORMAP_JET);
    cv::addWeighted(frame, 0.7, colored, 0.3, 0.0, frame);
}

}  // namespace

void Visualizer::render(cv::Mat& frame, const pipeline::FrameOutput& out) const {
    if (frame.empty()) return;

    draw_heatmap(frame, out.fused_heatmap);

    // Zone outlines + per-zone score.
    for (const auto& [hit, score] : out.zone_scores) {
        // We don't have the polygon here without going back to the
        // ROI manager; skip outline drawing for brevity. The fused
        // heatmap already conveys the spatial info.
        (void)hit;
        (void)score;
    }

    // Status banner.
    cv::Scalar banner_color = cv::Scalar(120, 120, 120);
    std::string status = "OK";
    if (out.firing) {
        status = "ANOMALY";
        if (!out.alerts.empty()) {
            banner_color = severity_color(out.alerts.front().severity);
        } else {
            banner_color = cv::Scalar(40, 165, 240);
        }
    }
    cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 32), banner_color, cv::FILLED);

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s  fused=%.3f  smoothed=%.3f  alerts=%zu",
                  status.c_str(),
                  static_cast<double>(out.fused_score),
                  static_cast<double>(out.smoothed_score), out.alerts.size());
    cv::putText(frame, buf, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);
}

}  // namespace eanom::overlay
