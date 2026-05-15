#include "eanom/ensemble/score_fusion.hpp"

#include <algorithm>
#include <cmath>

namespace eanom::ensemble {

namespace {

/// Squash any non-negative score into [0, 1) so that detectors with
/// wildly different scales (e.g. PaDiM Mahalanobis ~30, motion area
/// ratio ~0.05) play together.
float normalise(float score) {
    return 1.0f - std::exp(-std::max(0.0f, score));
}

}  // namespace

ScoreFusion::ScoreFusion(const config::EnsembleConfig& cfg) : cfg_(cfg) {}

float ScoreFusion::fuse(const std::vector<detectors::DetectionResult>& results) const {
    if (results.empty()) return 0.0f;

    if (cfg_.fusion == "max") {
        float best = 0.0f;
        for (const auto& r : results) {
            const auto it = cfg_.weights.find(r.detector_name);
            if (it == cfg_.weights.end()) continue;
            const float v = it->second * normalise(r.score);
            best = std::max(best, v);
        }
        return best;
    }

    // Default: weighted sum
    float total = 0.0f;
    for (const auto& r : results) {
        const auto it = cfg_.weights.find(r.detector_name);
        if (it == cfg_.weights.end()) continue;
        total += it->second * normalise(r.score);
    }
    return total;
}

}  // namespace eanom::ensemble
