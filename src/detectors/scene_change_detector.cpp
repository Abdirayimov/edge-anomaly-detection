#include "eanom/detectors/scene_change_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <bitset>
#include <stdexcept>

namespace eanom::detectors {

SceneChangeDetector::SceneChangeDetector(const config::SceneChangeParams& params)
    : params_(params) {}

cv::Mat SceneChangeDetector::compute_hsv_histogram(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    const int h_bins = 32;
    const int s_bins = 32;
    const int hist_size[] = {h_bins, s_bins};
    const float h_ranges[] = {0.0f, 180.0f};
    const float s_ranges[] = {0.0f, 256.0f};
    const float* ranges[] = {h_ranges, s_ranges};
    const int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

std::uint64_t SceneChangeDetector::compute_phash(const cv::Mat& frame) {
    // Classic DCT perceptual hash: 32x32 grayscale -> DCT -> top-left
    // 8x8 (DC excluded) -> compare against median -> 64-bit hash.
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
    gray.convertTo(gray, CV_32F);
    cv::Mat dct;
    cv::dct(gray, dct);

    cv::Mat block = dct(cv::Rect(0, 0, 8, 8)).clone();
    // Exclude DC (top-left) when computing median to avoid being
    // dominated by mean brightness.
    std::vector<float> values;
    values.reserve(63);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (r == 0 && c == 0) continue;
            values.push_back(block.at<float>(r, c));
        }
    }
    std::nth_element(values.begin(), values.begin() + 31, values.end());
    const float median = values[31];

    std::uint64_t hash = 0;
    int bit = 0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (block.at<float>(r, c) > median) {
                hash |= (std::uint64_t{1} << bit);
            }
            ++bit;
        }
    }
    return hash;
}

float SceneChangeDetector::compute_edge_density(const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    const double total = static_cast<double>(edges.rows) * static_cast<double>(edges.cols);
    if (total <= 0.0) return 0.0f;
    return static_cast<float>(cv::countNonZero(edges)) / static_cast<float>(total);
}

std::uint32_t SceneChangeDetector::hamming_distance(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint32_t>(std::bitset<64>(a ^ b).count());
}

DetectionResult SceneChangeDetector::process(const cv::Mat& frame) {
    DetectionResult r;
    r.detector_name = "scene_change";
    if (frame.empty()) return r;

    const cv::Mat hist = compute_hsv_histogram(frame);
    const std::uint64_t phash = compute_phash(frame);
    const float edge_density = compute_edge_density(frame);

    float hist_norm = 0.0f;
    float phash_norm = 0.0f;
    float edge_norm = 0.0f;

    if (prev_hist_.has_value()) {
        int method = cv::HISTCMP_CHISQR;
        if (params_.histogram_method == "bhattacharyya") method = cv::HISTCMP_BHATTACHARYYA;
        else if (params_.histogram_method == "correl") method = cv::HISTCMP_CORREL;
        double d = cv::compareHist(*prev_hist_, hist, method);
        if (method == cv::HISTCMP_CORREL) d = 1.0 - d;  // correl: 1 = identical
        // Clamp into a 0..1 range; large chi-square values saturate.
        hist_norm = static_cast<float>(std::min(1.0, std::max(0.0, d / 5.0)));
    }
    if (prev_phash_.has_value()) {
        const auto hd = hamming_distance(*prev_phash_, phash);
        phash_norm = std::min(1.0f, static_cast<float>(hd) /
                                        static_cast<float>(params_.phash_threshold));
    }
    if (prev_edge_density_.has_value()) {
        const float delta = std::abs(edge_density - *prev_edge_density_);
        edge_norm = std::min(1.0f, delta / params_.edge_density_delta);
    }

    prev_hist_ = hist;
    prev_phash_ = phash;
    prev_edge_density_ = edge_density;

    r.score = std::max({hist_norm, phash_norm, edge_norm});
    // No spatially-resolved output; leave heatmap empty.
    return r;
}

}  // namespace eanom::detectors
