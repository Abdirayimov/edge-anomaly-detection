#include "eanom/pipeline/anomaly_pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include "eanom/utils/logger.hpp"

namespace eanom::pipeline {

namespace {

void accumulate_heatmap(cv::Mat& acc, const cv::Mat& src, float weight) {
    if (src.empty()) return;
    cv::Mat resized = src;
    if (src.size() != acc.size()) {
        cv::resize(src, resized, acc.size(), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat scaled;
    resized.convertTo(scaled, CV_32F, weight);
    if (acc.empty()) {
        acc = scaled;
    } else {
        acc = acc + scaled;
    }
}

}  // namespace

AnomalyPipeline::AnomalyPipeline(const config::SystemConfig& cfg) : cfg_(cfg) {
    if (cfg_.detectors.motion.enabled) {
        motion_ = std::make_unique<detectors::MotionDetector>(cfg_.detectors.motion);
    }
    if (cfg_.detectors.scene_change.enabled) {
        scene_ = std::make_unique<detectors::SceneChangeDetector>(cfg_.detectors.scene_change);
    }
#ifdef EANOM_HAVE_TRT
    if (cfg_.detectors.padim.enabled) {
        padim_ = std::make_unique<detectors::PaDiMDetector>(cfg_.detectors.padim);
    }
#endif
    fusion_ = std::make_unique<ensemble::ScoreFusion>(cfg_.ensemble);
    smoother_ = std::make_unique<ensemble::TemporalSmoother>(cfg_.ensemble);
    roi_ = std::make_unique<roi::RoiManager>(
        cfg_.roi, static_cast<int>(cfg_.pipeline.input_width),
        static_cast<int>(cfg_.pipeline.input_height));
    alerts_ = std::make_unique<events::AlertGenerator>(cfg_.events);
    for (const auto& sink_spec : cfg_.events.sinks) {
        if (auto sink = events::make_event_sink(sink_spec)) {
            alerts_->register_sink(std::move(sink));
        }
    }
}

FrameOutput AnomalyPipeline::process(std::uint64_t frame_number, const cv::Mat& frame) {
    FrameOutput out;
    out.frame_number = frame_number;
    if (frame.empty()) return out;

    if (motion_) out.per_detector.push_back(motion_->process(frame));
    if (scene_) out.per_detector.push_back(scene_->process(frame));
#ifdef EANOM_HAVE_TRT
    if (padim_) out.per_detector.push_back(padim_->process(frame));
#endif

    out.fused_score = fusion_->fuse(out.per_detector);

    // Build a frame-resolution fused heatmap as a weighted sum of
    // each detector's heatmap; detectors without a heatmap (scene
    // change) contribute nothing here. The fused heatmap is what the
    // visualizer renders and what the ROI manager scores against.
    cv::Mat acc;
    if (!frame.empty()) {
        acc = cv::Mat::zeros(frame.rows, frame.cols, CV_32F);
    }
    for (const auto& r : out.per_detector) {
        if (r.heatmap.empty()) continue;
        const auto wit = cfg_.ensemble.weights.find(r.detector_name);
        if (wit == cfg_.ensemble.weights.end()) continue;
        accumulate_heatmap(acc, r.heatmap, wit->second);
    }
    roi_->apply_ignore_mask(acc);
    out.fused_heatmap = acc;

    out.zone_scores = roi_->score_in_zones(acc);

    constexpr float kFiringThreshold = 0.4f;
    out.smoothed_score = smoother_->push(out.fused_score, kFiringThreshold);
    out.firing = smoother_->is_firing();

    if (out.firing) {
        // Fire one alert per non-ignored zone whose score exceeds the
        // firing threshold; zones without an ROI mask still fire under
        // the generic "frame" zone label.
        bool any_zone_alert = false;
        for (const auto& [hit, score] : out.zone_scores) {
            if (hit.ignored) continue;
            if (score <= 0.0f) continue;
            const auto a = alerts_->consider(frame_number, hit.zone_name, score, true);
            out.alerts.insert(out.alerts.end(), a.begin(), a.end());
            any_zone_alert = true;
        }
        if (!any_zone_alert) {
            const auto a = alerts_->consider(frame_number, "frame", out.smoothed_score, true);
            out.alerts.insert(out.alerts.end(), a.begin(), a.end());
        }
    }
    return out;
}

}  // namespace eanom::pipeline
