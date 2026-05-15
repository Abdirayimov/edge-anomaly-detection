#include "eanom/detectors/motion_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include <stdexcept>
#include <vector>

namespace eanom::detectors {

MotionDetector::MotionDetector(const config::MotionParams& params) : params_(params) {
    if (params_.type == "mog2") {
        bg_ = cv::createBackgroundSubtractorMOG2(
            params_.history, params_.var_threshold, params_.detect_shadows);
    } else if (params_.type == "knn") {
        bg_ = cv::createBackgroundSubtractorKNN(
            params_.history, params_.var_threshold, params_.detect_shadows);
    } else {
        throw std::invalid_argument("unknown motion detector type: " + params_.type);
    }
    morph_kernel_ = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(params_.morphology_kernel, params_.morphology_kernel));
}

DetectionResult MotionDetector::process(const cv::Mat& frame) {
    DetectionResult r;
    r.detector_name = "motion";
    if (frame.empty()) return r;

    cv::Mat fg;
    bg_->apply(frame, fg);

    // Drop shadows (graylevel 127 when detect_shadows is on) to a
    // hard binary mask.
    cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);

    // Morphological open removes speckle, dilate consolidates blobs.
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN, morph_kernel_);
    cv::dilate(fg, fg, morph_kernel_);

    // Connected component filtering: drop blobs smaller than threshold.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat keep_mask = cv::Mat::zeros(fg.size(), CV_8U);
    double total_kept_area = 0.0;
    for (const auto& c : contours) {
        const double a = cv::contourArea(c);
        if (a < params_.min_contour_area) continue;
        total_kept_area += a;
        cv::drawContours(keep_mask, std::vector<std::vector<cv::Point>>{c}, -1,
                         cv::Scalar(255), cv::FILLED);
    }

    const double frame_area = static_cast<double>(frame.rows) * static_cast<double>(frame.cols);
    r.score = (frame_area > 0.0) ? static_cast<float>(total_kept_area / frame_area) : 0.0f;

    // Heatmap: binary mask cast to float32 in [0, 1].
    keep_mask.convertTo(r.heatmap, CV_32F, 1.0 / 255.0);
    return r;
}

}  // namespace eanom::detectors
