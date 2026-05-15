#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "eanom/config/system_config.hpp"
#include "eanom/detectors/common.hpp"

namespace eanom::ensemble {

/// Fuse per-detector scores into a single scalar in [0, +inf).
///
/// Methods:
///   * "weighted_sum" - sum(weight[name] * normalised(score)). The
///     weights need not sum to 1; the comparator downstream uses an
///     absolute threshold.
///   * "max" - max over (weight[name] * normalised(score)). Most
///     conservative: any single detector firing high produces a high
///     fused score.
class ScoreFusion {
public:
    explicit ScoreFusion(const config::EnsembleConfig& cfg);

    /// Fuse a frame's worth of detector outputs. Detectors absent
    /// from `weights` are ignored.
    float fuse(const std::vector<detectors::DetectionResult>& results) const;

    const config::EnsembleConfig& config() const noexcept { return cfg_; }

private:
    config::EnsembleConfig cfg_;
};

}  // namespace eanom::ensemble
