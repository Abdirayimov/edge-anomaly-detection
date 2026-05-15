#include "eanom/config/system_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace eanom::config {

namespace {

template <typename T>
T require(const YAML::Node& node, const std::string& key) {
    if (!node[key]) throw std::runtime_error("missing required config key: " + key);
    return node[key].as<T>();
}

template <typename T>
T optional(const YAML::Node& node, const std::string& key, T fallback) {
    return node[key] ? node[key].as<T>() : fallback;
}

}  // namespace

SystemConfig SystemConfig::load(const std::string& yaml_path) {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    SystemConfig out;

    if (const auto p = root["pipeline"]; p) {
        out.pipeline.input_width = optional<std::uint32_t>(p, "input_width", 1280);
        out.pipeline.input_height = optional<std::uint32_t>(p, "input_height", 720);
        out.pipeline.emit_overlay = optional<bool>(p, "emit_overlay", true);
    }

    if (const auto d = root["detectors"]; d) {
        if (const auto m = d["motion"]; m) {
            out.detectors.motion.enabled = optional<bool>(m, "enabled", true);
            out.detectors.motion.type = optional<std::string>(m, "type", "mog2");
            out.detectors.motion.history = optional<int>(m, "history", 500);
            out.detectors.motion.var_threshold = optional<double>(m, "var_threshold", 16.0);
            out.detectors.motion.detect_shadows = optional<bool>(m, "detect_shadows", false);
            out.detectors.motion.min_contour_area = optional<int>(m, "min_contour_area", 500);
            out.detectors.motion.morphology_kernel = optional<int>(m, "morphology_kernel", 5);
        }
        if (const auto s = d["scene_change"]; s) {
            out.detectors.scene_change.enabled = optional<bool>(s, "enabled", true);
            out.detectors.scene_change.histogram_method =
                optional<std::string>(s, "histogram_method", "chisqr");
            out.detectors.scene_change.phash_threshold =
                optional<int>(s, "phash_threshold", 8);
            out.detectors.scene_change.edge_density_delta =
                optional<float>(s, "edge_density_delta", 0.05f);
        }
        if (const auto pa = d["padim"]; pa) {
            out.detectors.padim.enabled = optional<bool>(pa, "enabled", false);
            out.detectors.padim.backbone_engine_path =
                optional<std::string>(pa, "backbone_engine_path", "");
            out.detectors.padim.stats_path = optional<std::string>(pa, "stats_path", "");
            out.detectors.padim.input_width = optional<std::uint32_t>(pa, "input_width", 224);
            out.detectors.padim.input_height = optional<std::uint32_t>(pa, "input_height", 224);
            out.detectors.padim.feature_dim = optional<std::uint32_t>(pa, "feature_dim", 100);
            out.detectors.padim.threshold = optional<float>(pa, "threshold", 5.0f);
        }
    }

    if (const auto e = root["ensemble"]; e) {
        out.ensemble.fusion = optional<std::string>(e, "fusion", "weighted_sum");
        if (const auto w = e["weights"]; w && w.IsMap()) {
            out.ensemble.weights.clear();
            for (const auto& kv : w) {
                out.ensemble.weights[kv.first.as<std::string>()] = kv.second.as<float>();
            }
        }
        if (const auto t = e["temporal"]; t) {
            out.ensemble.ema_alpha = optional<float>(t, "ema_alpha", 0.3f);
            out.ensemble.min_duration_frames =
                optional<std::uint32_t>(t, "min_duration_frames", 5);
        }
    }

    if (const auto r = root["roi"]; r) {
        if (const auto zs = r["zones"]; zs && zs.IsSequence()) {
            for (const auto& z : zs) {
                ZoneEntry e;
                e.name = require<std::string>(z, "name");
                e.ignore = optional<bool>(z, "ignore", false);
                e.severity_multiplier = optional<float>(z, "severity_multiplier", 1.0f);
                if (const auto poly = z["polygon"]; poly && poly.IsSequence()) {
                    for (const auto& pt : poly) {
                        if (pt.IsSequence() && pt.size() == 2) {
                            e.polygon.emplace_back(pt[0].as<int>(), pt[1].as<int>());
                        }
                    }
                }
                out.roi.zones.push_back(std::move(e));
            }
        }
    }

    if (const auto ev = root["events"]; ev) {
        out.events.cooldown_seconds = optional<std::uint32_t>(ev, "cooldown_seconds", 30);
        if (const auto sinks = ev["sinks"]; sinks && sinks.IsSequence()) {
            out.events.sinks.clear();
            for (const auto& s : sinks) {
                EventSinkSpec spec;
                spec.type = require<std::string>(s, "type");
                spec.path = optional<std::string>(s, "path", "");
                spec.url = optional<std::string>(s, "url", "");
                out.events.sinks.push_back(std::move(spec));
            }
        }
    }

    if (const auto l = root["logging"]; l) {
        out.logging.level = optional<std::string>(l, "level", "info");
        out.logging.json = optional<bool>(l, "json", false);
    }

    return out;
}

}  // namespace eanom::config
