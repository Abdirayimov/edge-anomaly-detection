#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>

#include "eanom/config/system_config.hpp"
#include "eanom/detectors/common.hpp"

namespace eanom::detectors {

/// Scene-change detector: looks for *global* differences between the
/// current frame and a recent reference (the previous frame).
///
/// Combines three orthogonal signals:
///   1. **HSV histogram distance.** Catches lighting / colour shifts
///      (someone turned on the lights, a camera switched its
///      white-balance preset).
///   2. **Perceptual hash (pHash) Hamming distance.** Catches scene
///      composition changes (camera was repositioned, a large object
///      appeared or moved).
///   3. **Edge density delta.** Catches structural changes the
///      histogram misses (a curtain came down, a door opened) and
///      provides a tie-breaker when histogram + pHash disagree.
///
/// Scalar score is `max(hist_norm, phash_norm, edge_norm)` so that any
/// single signal can flag a scene change on its own.
class SceneChangeDetector {
public:
    explicit SceneChangeDetector(const config::SceneChangeParams& params);

    DetectionResult process(const cv::Mat& frame);

    const config::SceneChangeParams& params() const noexcept { return params_; }

private:
    config::SceneChangeParams params_;
    std::optional<cv::Mat> prev_hist_;
    std::optional<std::uint64_t> prev_phash_;
    std::optional<float> prev_edge_density_;

    static cv::Mat compute_hsv_histogram(const cv::Mat& frame);
    static std::uint64_t compute_phash(const cv::Mat& frame);
    static float compute_edge_density(const cv::Mat& frame);
    static std::uint32_t hamming_distance(std::uint64_t a, std::uint64_t b);
};

}  // namespace eanom::detectors
