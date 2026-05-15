#include "eanom/roi/roi_manager.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace eanom::roi {

RoiManager::RoiManager(const config::RoiConfig& cfg, int frame_width, int frame_height)
    : frame_width_(frame_width), frame_height_(frame_height), zones_(cfg.zones) {
    ignore_mask_ = cv::Mat::zeros(frame_height_, frame_width_, CV_8U);
    zone_masks_.reserve(zones_.size());
    for (const auto& z : zones_) {
        cv::Mat mask = cv::Mat::zeros(frame_height_, frame_width_, CV_8U);
        if (!z.polygon.empty()) {
            std::vector<std::vector<cv::Point>> polys{z.polygon};
            cv::fillPoly(mask, polys, cv::Scalar(255));
            if (z.ignore) {
                cv::bitwise_or(ignore_mask_, mask, ignore_mask_);
            }
        }
        zone_masks_.push_back(std::move(mask));
    }
}

void RoiManager::apply_ignore_mask(cv::Mat& heatmap) const {
    if (heatmap.empty() || ignore_mask_.empty()) return;
    cv::Mat ignore_resized = ignore_mask_;
    if (heatmap.size() != ignore_mask_.size()) {
        cv::resize(ignore_mask_, ignore_resized, heatmap.size(), 0, 0, cv::INTER_NEAREST);
    }
    heatmap.setTo(cv::Scalar(0), ignore_resized);
}

std::vector<std::pair<ZoneHit, float>> RoiManager::score_in_zones(
    const cv::Mat& heatmap) const {
    std::vector<std::pair<ZoneHit, float>> out;
    out.reserve(zones_.size());
    if (heatmap.empty()) {
        for (const auto& z : zones_) {
            ZoneHit h{z.name, z.severity_multiplier, z.ignore};
            out.emplace_back(std::move(h), 0.0f);
        }
        return out;
    }
    for (std::size_t i = 0; i < zones_.size(); ++i) {
        const auto& z = zones_[i];
        ZoneHit h{z.name, z.severity_multiplier, z.ignore};
        if (z.ignore) {
            out.emplace_back(std::move(h), 0.0f);
            continue;
        }
        cv::Mat mask = zone_masks_[i];
        if (mask.size() != heatmap.size()) {
            cv::resize(zone_masks_[i], mask, heatmap.size(), 0, 0, cv::INTER_NEAREST);
        }
        double max_v = 0.0;
        cv::minMaxLoc(heatmap, nullptr, &max_v, nullptr, nullptr, mask);
        const float score = static_cast<float>(max_v) * z.severity_multiplier;
        out.emplace_back(std::move(h), score);
    }
    return out;
}

}  // namespace eanom::roi
