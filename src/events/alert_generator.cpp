#include "eanom/events/alert_generator.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "eanom/utils/logger.hpp"

namespace eanom::events {

namespace {

class StdoutSink : public IEventSink {
public:
    void emit(const Alert& a) override {
        std::cout << "[" << severity_name(a.severity) << "] frame=" << a.frame_number
                  << " zone=\"" << a.zone << "\" score=" << a.score << "\n";
    }
};

class FileSink : public IEventSink {
public:
    explicit FileSink(const std::string& path) : path_(path) {}
    void emit(const Alert& a) override {
        std::ofstream f(path_, std::ios::app);
        if (!f.is_open()) {
            EANOM_LOG_WARN("alert: cannot append to {}", path_);
            return;
        }
        f << "{\"frame\":" << a.frame_number
          << ",\"severity\":\"" << severity_name(a.severity) << "\""
          << ",\"zone\":\"" << a.zone << "\""
          << ",\"score\":" << a.score << "}\n";
    }

private:
    std::string path_;
};

class WebhookSink : public IEventSink {
public:
    explicit WebhookSink(const std::string& url) : url_(url) {
        EANOM_LOG_INFO("webhook sink configured: {}", url_);
    }
    void emit(const Alert& a) override {
        // Reference implementation only logs the payload; production
        // would post via libcurl. Kept stub here so the toolkit does
        // not pull libcurl into the default link line.
        EANOM_LOG_INFO("webhook would POST: frame={} severity={} zone={} score={}",
                       a.frame_number, severity_name(a.severity), a.zone, a.score);
    }

private:
    std::string url_;
};

}  // namespace

std::unique_ptr<IEventSink> make_event_sink(const config::EventSinkSpec& spec) {
    if (spec.type == "stdout") return std::make_unique<StdoutSink>();
    if (spec.type == "file") return std::make_unique<FileSink>(spec.path);
    if (spec.type == "webhook") return std::make_unique<WebhookSink>(spec.url);
    EANOM_LOG_WARN("unknown event sink type: {}", spec.type);
    return nullptr;
}

AlertGenerator::AlertGenerator(const config::EventsConfig& cfg) : cfg_(cfg) {}

void AlertGenerator::register_sink(std::unique_ptr<IEventSink> sink) {
    if (sink) sinks_.push_back(std::move(sink));
}

Severity AlertGenerator::classify_(float score) const {
    if (score >= 1.5f) return Severity::Critical;
    if (score >= 0.7f) return Severity::Warning;
    return Severity::Info;
}

std::vector<Alert> AlertGenerator::consider(std::uint64_t frame_number,
                                             const std::string& zone, float score,
                                             bool firing) {
    std::vector<Alert> out;
    if (!firing || score <= 0.0f) return out;

    const Severity sev = classify_(score);
    const auto now = std::chrono::steady_clock::now();
    const auto cooldown = std::chrono::seconds(cfg_.cooldown_seconds);

    auto& last = last_fires_[zone];
    if (last.ts.time_since_epoch().count() != 0 && (now - last.ts) < cooldown &&
        sev <= last.severity) {
        return out;
    }
    last.ts = now;
    last.severity = sev;

    Alert a;
    a.timestamp = now;
    a.severity = sev;
    a.zone = zone;
    a.score = score;
    a.frame_number = frame_number;

    for (auto& sink : sinks_) sink->emit(a);
    out.push_back(a);
    return out;
}

}  // namespace eanom::events
