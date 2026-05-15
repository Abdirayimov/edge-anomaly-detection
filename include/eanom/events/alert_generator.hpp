#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "eanom/config/system_config.hpp"

namespace eanom::events {

enum class Severity {
    Info,
    Warning,
    Critical,
};

inline const char* severity_name(Severity s) noexcept {
    switch (s) {
        case Severity::Info:     return "info";
        case Severity::Warning:  return "warning";
        case Severity::Critical: return "critical";
    }
    return "?";
}

struct Alert {
    std::chrono::steady_clock::time_point timestamp;
    Severity severity = Severity::Info;
    std::string zone;
    float score = 0.0f;
    std::uint64_t frame_number = 0;
};

/// Pluggable alert sink (stdout / file / webhook).
class IEventSink {
public:
    virtual ~IEventSink() = default;
    virtual void emit(const Alert& alert) = 0;
};

std::unique_ptr<IEventSink> make_event_sink(const config::EventSinkSpec& spec);

/// Generates alerts subject to cooldown + severity laddering.
///
/// Calling `consider(...)` once per frame is the expected usage.
/// Alerts only fire when:
///   * the smoothed/zone-resolved score crosses the next severity
///     boundary, AND
///   * a cooldown of `cooldown_seconds` has elapsed since the
///     previous fire for that zone at that severity.
class AlertGenerator {
public:
    AlertGenerator(const config::EventsConfig& cfg);

    /// Consider the current frame. Returns any alerts that should be
    /// emitted right now; emits them through every registered sink.
    std::vector<Alert> consider(std::uint64_t frame_number, const std::string& zone,
                                float score, bool firing);

    void register_sink(std::unique_ptr<IEventSink> sink);

private:
    config::EventsConfig cfg_;
    std::vector<std::unique_ptr<IEventSink>> sinks_;
    struct LastFire {
        std::chrono::steady_clock::time_point ts{};
        Severity severity = Severity::Info;
    };
    std::unordered_map<std::string, LastFire> last_fires_;

    Severity classify_(float score) const;
};

}  // namespace eanom::events
