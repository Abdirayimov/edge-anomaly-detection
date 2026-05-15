#pragma once

#include <spdlog/spdlog.h>

#include <string>

namespace eanom::utils {

void init_logger(const std::string& level, bool json);

}  // namespace eanom::utils

#define EANOM_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define EANOM_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define EANOM_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define EANOM_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define EANOM_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
