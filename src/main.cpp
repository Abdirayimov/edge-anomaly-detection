// eanom_video: OpenCV-based offline anomaly-detection driver.

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "eanom/config/system_config.hpp"
#include "eanom/overlay/visualizer.hpp"
#include "eanom/pipeline/anomaly_pipeline.hpp"
#include "eanom/utils/logger.hpp"

namespace {

std::atomic<bool> g_shutdown{false};
void sig(int) { g_shutdown = true; }

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --config CONFIG_YAML --input VIDEO [--output ANNOTATED.mp4]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](const std::string& flag) {
            if (i + 1 >= argc) throw std::invalid_argument(flag + " expects a value");
            return std::string(argv[++i]);
        };
        if (a == "--config" || a == "-c") config_path = take(a);
        else if (a == "--input" || a == "-i") input_path = take(a);
        else if (a == "--output" || a == "-o") output_path = take(a);
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty() || input_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, sig);
    std::signal(SIGTERM, sig);

    try {
        const auto cfg = eanom::config::SystemConfig::load(config_path);
        eanom::utils::init_logger(cfg.logging.level, cfg.logging.json);

        eanom::pipeline::AnomalyPipeline pipe(cfg);
        eanom::overlay::Visualizer viz;

        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("could not open input video: " + input_path);
        }
        const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);

        std::unique_ptr<cv::VideoWriter> writer;
        if (!output_path.empty()) {
            writer = std::make_unique<cv::VideoWriter>(
                output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                (fps > 0.0 ? fps : 25.0), cv::Size(W, H));
            if (!writer->isOpened()) {
                throw std::runtime_error("could not open output: " + output_path);
            }
        }

        std::uint64_t frame_no = 0;
        cv::Mat frame;
        std::uint64_t total_alerts = 0;
        while (cap.read(frame)) {
            if (g_shutdown.load()) break;
            const auto out = pipe.process(frame_no++, frame);
            total_alerts += out.alerts.size();
            if (writer) {
                viz.render(frame, out);
                writer->write(frame);
            }
            if (frame_no % 60 == 0) {
                SPDLOG_INFO("frame={} fused={:.3f} firing={} alerts_total={}", frame_no,
                            out.fused_score, out.firing, total_alerts);
            }
        }
        SPDLOG_INFO("done: {} frames processed, {} alerts emitted", frame_no, total_alerts);
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
