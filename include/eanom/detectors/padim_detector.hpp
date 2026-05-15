#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "eanom/config/system_config.hpp"
#include "eanom/detectors/common.hpp"

namespace eanom::trt {
class TrtEngine;
}

namespace eanom::detectors {

/// On-disk PaDiM statistics produced by training/fit_padim.py.
///
/// Layout (little-endian):
///   uint32  magic = 0x50414449 ('PADI')
///   uint32  version = 1
///   uint32  H (feature map height after spatial reduction)
///   uint32  W
///   uint32  D (channel dim after random-subset projection)
///   uint32  projection_dim_full (D' the backbone produced before subset)
///   uint32[D] selected_channel_indices
///   float32[H*W*D] means
///   float32[H*W*D*D] inv_covariances
///
/// For H=W=28, D=100, file ~= 31 MB.
struct PaDiMStats {
    std::uint32_t H = 0;
    std::uint32_t W = 0;
    std::uint32_t D = 0;
    std::uint32_t projection_dim_full = 0;
    std::vector<std::uint32_t> selected_channels;
    std::vector<float> means;          ///< H*W*D
    std::vector<float> inv_covariances;  ///< H*W*D*D

    static PaDiMStats load(const std::string& path);
};

/// PaDiM (Defard et al., 2020) feature-distribution anomaly detector.
///
/// On each frame:
///   1. Resize to backbone input size; ImageNet-normalise.
///   2. Run backbone forward; extract feature map (D' channels, H*W).
///   3. Project to D channels using `selected_channels` from stats.
///   4. For each (h, w): compute Mahalanobis distance using the
///      stored mean[h, w] and inv_covariance[h, w]:
///          d^2 = (f - mu)^T inv_cov (f - mu)
///   5. Heatmap = sqrt(d^2) at HxW; score = max (or 99th percentile).
class PaDiMDetector {
public:
    explicit PaDiMDetector(const config::PaDiMParams& params);
    ~PaDiMDetector();

    PaDiMDetector(const PaDiMDetector&) = delete;
    PaDiMDetector& operator=(const PaDiMDetector&) = delete;

    DetectionResult process(const cv::Mat& frame);

    const config::PaDiMParams& params() const noexcept { return params_; }

private:
    config::PaDiMParams params_;
    PaDiMStats stats_;
    std::unique_ptr<trt::TrtEngine> backbone_;
    std::vector<float> input_scratch_;
    std::vector<float> feature_scratch_;

    float mahalanobis_squared_(const float* feature, std::uint32_t h, std::uint32_t w) const;
};

}  // namespace eanom::detectors
