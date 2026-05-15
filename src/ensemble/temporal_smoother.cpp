#include "eanom/ensemble/temporal_smoother.hpp"

namespace eanom::ensemble {

TemporalSmoother::TemporalSmoother(const config::EnsembleConfig& cfg) : cfg_(cfg) {}

float TemporalSmoother::push(float raw_score, float threshold) {
    if (!initialised_) {
        smoothed_ = raw_score;
        initialised_ = true;
    } else {
        smoothed_ = cfg_.ema_alpha * raw_score + (1.0f - cfg_.ema_alpha) * smoothed_;
    }
    if (smoothed_ > threshold) {
        ++above_;
    } else {
        above_ = 0;
    }
    return smoothed_;
}

bool TemporalSmoother::is_firing() const noexcept {
    return above_ >= cfg_.min_duration_frames;
}

}  // namespace eanom::ensemble
