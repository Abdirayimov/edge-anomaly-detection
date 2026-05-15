#pragma once

#include <opencv2/core.hpp>

#include "eanom/pipeline/anomaly_pipeline.hpp"

namespace eanom::overlay {

/// Draw the fused heatmap, ROI zone outlines, score readout, and
/// alert badge on top of an input frame.
class Visualizer {
public:
    /// Render in place. `frame` must be CV_8UC3.
    void render(cv::Mat& frame, const pipeline::FrameOutput& out) const;
};

}  // namespace eanom::overlay
