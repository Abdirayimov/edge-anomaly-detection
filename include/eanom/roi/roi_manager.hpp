#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "eanom/config/system_config.hpp"

namespace eanom::roi {

/// Resolved ROI hit: the zone where an event would fire.
struct ZoneHit {
    std::string zone_name;
    float severity_multiplier = 1.0f;
    bool ignored = false;
};

/// Polygonal ROI manager.
///
/// On construction, rasterises every configured zone into an integer
/// mask (one mask channel per zone) and an `ignore` mask. At runtime,
/// `score_in_zones()` takes a heatmap and returns the maximum score
/// inside each non-ignored zone, weighted by that zone's severity
/// multiplier.
///
/// "Ignore" zones (e.g. a road, an HVAC fan, the camera's date
/// overlay) are subtracted from every detection mask before scoring
/// to suppress nuisance alerts.
class RoiManager {
public:
    RoiManager(const config::RoiConfig& cfg, int frame_width, int frame_height);

    /// Apply ignore masks to a per-pixel heatmap in place. Pixels
    /// inside any ignore zone are set to zero.
    void apply_ignore_mask(cv::Mat& heatmap) const;

    /// Compute the maximum heatmap value inside each zone (scaled by
    /// severity_multiplier). Returns one ZoneHit per zone, in
    /// configuration order. Ignored zones report 0.0f.
    std::vector<std::pair<ZoneHit, float>> score_in_zones(const cv::Mat& heatmap) const;

    bool empty() const noexcept { return zones_.empty(); }
    int frame_width() const noexcept { return frame_width_; }
    int frame_height() const noexcept { return frame_height_; }

    const std::vector<config::ZoneEntry>& zones() const noexcept { return zones_; }

private:
    int frame_width_;
    int frame_height_;
    std::vector<config::ZoneEntry> zones_;
    std::vector<cv::Mat> zone_masks_;   ///< CV_8U per zone
    cv::Mat ignore_mask_;                ///< CV_8U, union of all ignore zones
};

}  // namespace eanom::roi
