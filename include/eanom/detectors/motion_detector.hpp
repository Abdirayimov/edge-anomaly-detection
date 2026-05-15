#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>

#include <memory>

#include "eanom/config/system_config.hpp"
#include "eanom/detectors/common.hpp"

namespace eanom::detectors {

/// Motion-based anomaly via background subtraction + connected-
/// component filtering.
///
/// Uses OpenCV's MOG2 or KNN background segmenter. Each frame:
///   1. Apply background subtractor to get a foreground mask.
///   2. Morphological open + dilate to clean up speckle and merge
///      adjacent blobs.
///   3. Find connected components; discard any whose area is below
///      `min_contour_area`.
///   4. score = sum of surviving blob areas / frame area, in [0, 1].
///      heatmap = the binary mask cast to float32.
///
/// Detector is intentionally cheap; it runs first in the ensemble and
/// the downstream PaDiM stage can be skipped when motion is below a
/// gating threshold (a future optimisation).
class MotionDetector {
public:
    explicit MotionDetector(const config::MotionParams& params);

    DetectionResult process(const cv::Mat& frame);

    const config::MotionParams& params() const noexcept { return params_; }

private:
    config::MotionParams params_;
    cv::Ptr<cv::BackgroundSubtractor> bg_;
    cv::Mat morph_kernel_;
};

}  // namespace eanom::detectors
