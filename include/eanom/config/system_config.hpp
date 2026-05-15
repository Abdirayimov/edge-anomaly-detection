#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace eanom::config {

// ---- per-detector parameter blocks ----

struct MotionParams {
    bool enabled = true;
    std::string type = "mog2";      // "mog2" | "knn"
    int history = 500;
    double var_threshold = 16.0;
    bool detect_shadows = false;
    int min_contour_area = 500;
    int morphology_kernel = 5;
};

struct SceneChangeParams {
    bool enabled = true;
    std::string histogram_method = "chisqr";  // chisqr | bhattacharyya | correl
    int phash_threshold = 8;
    float edge_density_delta = 0.05f;
};

struct PaDiMParams {
    bool enabled = false;
    std::string backbone_engine_path;
    std::string stats_path;
    std::uint32_t input_width = 224;
    std::uint32_t input_height = 224;
    std::uint32_t feature_dim = 100;
    float threshold = 5.0f;
};

struct DetectorsConfig {
    MotionParams motion;
    SceneChangeParams scene_change;
    PaDiMParams padim;
};

// ---- ensemble + smoothing ----

struct EnsembleConfig {
    std::string fusion = "weighted_sum";
    std::unordered_map<std::string, float> weights{
        {"motion", 0.3f}, {"scene_change", 0.2f}, {"padim", 0.5f}};
    float ema_alpha = 0.3f;
    std::uint32_t min_duration_frames = 5;
};

// ---- ROI ----

struct ZoneEntry {
    std::string name;
    std::vector<cv::Point> polygon;
    bool ignore = false;
    float severity_multiplier = 1.0f;
};

struct RoiConfig {
    std::vector<ZoneEntry> zones;
};

// ---- events ----

struct EventSinkSpec {
    std::string type;                       // "stdout" | "file" | "webhook"
    std::string path;                       // file sinks
    std::string url;                        // webhook sinks
};

struct EventsConfig {
    std::uint32_t cooldown_seconds = 30;
    std::vector<std::string> severity_levels{"info", "warning", "critical"};
    std::vector<EventSinkSpec> sinks{{"stdout", "", ""}};
};

// ---- pipeline + logging ----

struct PipelineConfig {
    std::uint32_t input_width = 1280;
    std::uint32_t input_height = 720;
    bool emit_overlay = true;
};

struct LoggingConfig {
    std::string level = "info";
    bool json = false;
};

struct SystemConfig {
    PipelineConfig pipeline;
    DetectorsConfig detectors;
    EnsembleConfig ensemble;
    RoiConfig roi;
    EventsConfig events;
    LoggingConfig logging;

    static SystemConfig load(const std::string& yaml_path);
};

}  // namespace eanom::config
