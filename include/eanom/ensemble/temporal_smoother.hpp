#pragma once

#include <cstdint>

#include "eanom/config/system_config.hpp"

namespace eanom::ensemble {

/// EMA smoother + min-duration debouncer.
///
/// Raw frame-level fused scores are noisy; transient flickers (a leaf
/// crossing a camera, a brief lighting change) flag as anomalies on a
/// single frame and then disappear. This smoother:
///
///   1. Applies an exponential moving average with alpha `ema_alpha`
///      to produce a stable score.
///   2. Tracks how many consecutive frames the smoothed score has
///      stayed above the user threshold. The `is_firing()` accessor
///      returns true only after that count exceeds
///      `min_duration_frames`.
class TemporalSmoother {
public:
    explicit TemporalSmoother(const config::EnsembleConfig& cfg);

    /// Push a new fused score. Returns the smoothed value.
    float push(float raw_score, float threshold);

    /// True when the smoothed score has stayed above `threshold` for
    /// at least `min_duration_frames` consecutive frames.
    bool is_firing() const noexcept;

    float smoothed() const noexcept { return smoothed_; }
    std::uint32_t frames_above_threshold() const noexcept { return above_; }

private:
    config::EnsembleConfig cfg_;
    float smoothed_ = 0.0f;
    bool initialised_ = false;
    std::uint32_t above_ = 0;
};

}  // namespace eanom::ensemble
