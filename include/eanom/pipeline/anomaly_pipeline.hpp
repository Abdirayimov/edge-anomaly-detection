#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "eanom/config/system_config.hpp"
#include "eanom/detectors/common.hpp"
#include "eanom/detectors/motion_detector.hpp"
#include "eanom/detectors/scene_change_detector.hpp"
#include "eanom/ensemble/score_fusion.hpp"
#include "eanom/ensemble/temporal_smoother.hpp"
#include "eanom/events/alert_generator.hpp"
#include "eanom/roi/roi_manager.hpp"

#ifdef EANOM_HAVE_TRT
#include "eanom/detectors/padim_detector.hpp"
#endif

namespace eanom::pipeline {

struct FrameOutput {
    std::uint64_t frame_number = 0;
    std::vector<detectors::DetectionResult> per_detector;
    float fused_score = 0.0f;
    float smoothed_score = 0.0f;
    bool firing = false;
    std::vector<std::pair<roi::ZoneHit, float>> zone_scores;
    std::vector<events::Alert> alerts;
    cv::Mat fused_heatmap;   ///< CV_32F, frame-resolution
};

/// Single-stream anomaly pipeline.
///
/// Wires every detector + the ensemble + ROI + alerting together.
/// The pipeline is intentionally sequential so it runs predictably on
/// edge hardware; the optional PaDiM detector consumes most of the
/// budget when enabled.
class AnomalyPipeline {
public:
    explicit AnomalyPipeline(const config::SystemConfig& cfg);

    FrameOutput process(std::uint64_t frame_number, const cv::Mat& frame);

    const config::SystemConfig& config() const noexcept { return cfg_; }

private:
    config::SystemConfig cfg_;
    std::unique_ptr<detectors::MotionDetector> motion_;
    std::unique_ptr<detectors::SceneChangeDetector> scene_;
#ifdef EANOM_HAVE_TRT
    std::unique_ptr<detectors::PaDiMDetector> padim_;
#endif
    std::unique_ptr<ensemble::ScoreFusion> fusion_;
    std::unique_ptr<ensemble::TemporalSmoother> smoother_;
    std::unique_ptr<roi::RoiManager> roi_;
    std::unique_ptr<events::AlertGenerator> alerts_;
};

}  // namespace eanom::pipeline
