#pragma once
#include <iostream>
#include <string_view>

// Tiny leveled logger, no dependencies. Not on any hot path.
namespace neurort::log {

enum class Level { Error = 0, Warn, Info, Debug };

inline Level& level() {
  static Level l = Level::Info;
  return l;
}

inline void msg(Level lv, std::string_view tag, std::string_view m) {
  if (static_cast<int>(lv) <= static_cast<int>(level())) {
    std::cerr << '[' << tag << "] " << m << '\n';
  }
}

inline void error(std::string_view m) { msg(Level::Error, "error", m); }
inline void warn(std::string_view m) { msg(Level::Warn, "warn", m); }
inline void info(std::string_view m) { msg(Level::Info, "info", m); }
inline void debug(std::string_view m) { msg(Level::Debug, "debug", m); }

}  // namespace neurort::log
