#include "eanom/detectors/padim_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "eanom/trt/trt_engine.hpp"
#include "eanom/utils/cuda_helpers.hpp"
#include "eanom/utils/logger.hpp"

namespace eanom::detectors {

namespace {

constexpr std::uint32_t kPadimMagic = 0x50414449;  // 'PADI'

constexpr float kImageMean[3] = {123.675f, 116.28f, 103.53f};
constexpr float kImageStd[3] = {58.395f, 57.12f, 57.375f};

void preprocess(const cv::Mat& frame, int out_w, int out_h, float* dst) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    const int stride = out_w * out_h;
    for (int y = 0; y < out_h; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out_w; ++x) {
            const auto& px = row[x];
            const int idx = y * out_w + x;
            const float r = static_cast<float>(px[2]);
            const float g = static_cast<float>(px[1]);
            const float b = static_cast<float>(px[0]);
            dst[0 * stride + idx] = (r - kImageMean[0]) / kImageStd[0];
            dst[1 * stride + idx] = (g - kImageMean[1]) / kImageStd[1];
            dst[2 * stride + idx] = (b - kImageMean[2]) / kImageStd[2];
        }
    }
}

}  // namespace

PaDiMStats PaDiMStats::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open PaDiM stats: " + path);

    auto read_u32 = [&] {
        std::uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };

    const auto magic = read_u32();
    if (magic != kPadimMagic) {
        throw std::runtime_error("PaDiM stats: bad magic (got 0x" +
                                 std::to_string(magic) + ")");
    }
    const auto version = read_u32();
    if (version != 1) {
        throw std::runtime_error("PaDiM stats: unsupported version " +
                                 std::to_string(version));
    }

    PaDiMStats s;
    s.H = read_u32();
    s.W = read_u32();
    s.D = read_u32();
    s.projection_dim_full = read_u32();
    s.selected_channels.resize(s.D);
    f.read(reinterpret_cast<char*>(s.selected_channels.data()),
           static_cast<std::streamsize>(s.D * sizeof(std::uint32_t)));

    const std::size_t hw = static_cast<std::size_t>(s.H) * s.W;
    s.means.resize(hw * s.D);
    f.read(reinterpret_cast<char*>(s.means.data()),
           static_cast<std::streamsize>(s.means.size() * sizeof(float)));

    s.inv_covariances.resize(hw * s.D * s.D);
    f.read(reinterpret_cast<char*>(s.inv_covariances.data()),
           static_cast<std::streamsize>(s.inv_covariances.size() * sizeof(float)));

    if (!f) {
        throw std::runtime_error("PaDiM stats: short read on " + path);
    }
    return s;
}

PaDiMDetector::PaDiMDetector(const config::PaDiMParams& params) : params_(params) {
    stats_ = PaDiMStats::load(params_.stats_path);
    if (stats_.D != params_.feature_dim) {
        EANOM_LOG_WARN("PaDiMDetector: feature_dim mismatch (config={} vs stats={})",
                       params_.feature_dim, stats_.D);
    }
    backbone_ = std::make_unique<trt::TrtEngine>(params_.backbone_engine_path);
    input_scratch_.resize(static_cast<std::size_t>(3) * params_.input_height *
                          params_.input_width);
}

PaDiMDetector::~PaDiMDetector() = default;

float PaDiMDetector::mahalanobis_squared_(const float* feature, std::uint32_t h,
                                          std::uint32_t w) const {
    const std::size_t pos = static_cast<std::size_t>(h) * stats_.W + w;
    const float* mu = stats_.means.data() + pos * stats_.D;
    const float* inv_cov = stats_.inv_covariances.data() + pos * stats_.D * stats_.D;

    // diff = feature - mu; result = diff' * inv_cov * diff
    std::vector<float> diff(stats_.D);
    for (std::uint32_t i = 0; i < stats_.D; ++i) {
        diff[i] = feature[i] - mu[i];
    }
    // Two-pass matvec then dot.
    float total = 0.0f;
    for (std::uint32_t i = 0; i < stats_.D; ++i) {
        float row_sum = 0.0f;
        for (std::uint32_t j = 0; j < stats_.D; ++j) {
            row_sum += inv_cov[i * stats_.D + j] * diff[j];
        }
        total += diff[i] * row_sum;
    }
    return std::max(0.0f, total);
}

DetectionResult PaDiMDetector::process(const cv::Mat& frame) {
    DetectionResult r;
    r.detector_name = "padim";
    if (frame.empty()) return r;

    preprocess(frame, static_cast<int>(params_.input_width),
               static_cast<int>(params_.input_height), input_scratch_.data());

    const std::string input_name = backbone_->bindings().front().name;
    utils::CudaStream stream;
    backbone_->copy_input(input_name, input_scratch_.data(),
                          input_scratch_.size() * sizeof(float), stream.get());
    backbone_->infer(stream.get());

    // The backbone export is expected to emit one feature map binding of
    // shape (1, projection_dim_full, H, W). Pick the first non-input
    // binding to read.
    std::string out_name;
    for (const auto& b : backbone_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    if (out_name.empty()) {
        throw std::runtime_error("PaDiM backbone has no output binding");
    }
    const auto& ob = backbone_->binding(out_name);
    feature_scratch_.assign(ob.volume, 0.0f);
    backbone_->copy_output(out_name, feature_scratch_.data(),
                           feature_scratch_.size() * sizeof(float), stream.get());
    stream.synchronize();

    const std::uint32_t H = stats_.H;
    const std::uint32_t W = stats_.W;
    const std::uint32_t D = stats_.D;
    const std::uint32_t Dfull = stats_.projection_dim_full;

    // Project each (h, w) position from full feature dim to D using
    // selected_channels, then compute Mahalanobis distance.
    std::vector<float> projected(D);
    cv::Mat heatmap(static_cast<int>(H), static_cast<int>(W), CV_32F, cv::Scalar(0));

    float max_score = 0.0f;
    for (std::uint32_t h = 0; h < H; ++h) {
        for (std::uint32_t w = 0; w < W; ++w) {
            for (std::uint32_t d = 0; d < D; ++d) {
                const std::uint32_t src_c = stats_.selected_channels[d];
                const std::size_t feat_idx =
                    static_cast<std::size_t>(src_c) * H * W + h * W + w;
                projected[d] = feature_scratch_[feat_idx];
            }
            const float dist_sq = mahalanobis_squared_(projected.data(), h, w);
            const float dist = std::sqrt(dist_sq);
            heatmap.at<float>(static_cast<int>(h), static_cast<int>(w)) = dist;
            if (dist > max_score) max_score = dist;
        }
    }

    // Upsample heatmap to original frame resolution.
    cv::resize(heatmap, r.heatmap, frame.size(), 0, 0, cv::INTER_LINEAR);

    // Global score: max of the Mahalanobis distances normalised by
    // the configured threshold so the value is on the same [0..1+]
    // scale as the other detectors.
    r.score = max_score / std::max(1e-6f, params_.threshold);
    return r;
}

}  // namespace eanom::detectors

// silence -Wunused for placeholder when math.h not included
#include <cmath>
