cmake_minimum_required(VERSION 3.20)
project(NeuroRT LANGUAGES CXX)

# ---- Standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(CompilerWarnings)

# ---- Dependencies ----
find_package(OpenMP REQUIRED)

include(FetchContent)

# nlohmann/json — header-only JSON (config / IO seam to the Python frontend).
# Allow an offline override: -DNEURORT_JSON_DIR=/path/to/single-header-dir (containing nlohmann/json.hpp)
if(NEURORT_JSON_DIR)
  add_library(nlohmann_json INTERFACE)
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
  target_include_directories(nlohmann_json INTERFACE ${NEURORT_JSON_DIR})
else()
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# GoogleTest — unit/golden tests. Disable by -DNEURORT_BUILD_TESTS=OFF (e.g. offline).
option(NEURORT_BUILD_TESTS "Build the test suite" ON)
if(NEURORT_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

add_subdirectory(sim)
