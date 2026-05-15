#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace eanom::detectors {

/// One detector's per-frame output.
///
/// `score` is a scalar anomaly indicator in [0, +inf); higher means
/// more anomalous. Each detector defines its own scale; the ensemble
/// normalises before fusion.
///
/// `heatmap`, when non-empty, is a single-channel float32 image of the
/// same logical resolution as the input frame, where each pixel holds
/// the per-location anomaly contribution. Detectors that do not produce
/// spatially-resolved output (e.g. global histogram diff) leave it
/// empty.
struct DetectionResult {
    float score = 0.0f;
    cv::Mat heatmap;             ///< Optional CV_32F single channel
    std::string detector_name;
};

}  // namespace eanom::detectors
