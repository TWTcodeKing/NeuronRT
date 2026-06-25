# ---- Core simulator library ----
# Source files are appended here as each milestone sub-task lands.
add_library(neurort_core
  src/common/config.cpp
  src/functional/flit.cpp
  src/functional/router.cpp
  src/functional/mesh.cpp
  src/functional/sync_fsm.cpp
  src/functional/pe_stub.cpp
  src/functional/dendrite.cpp
  src/functional/dnp.cpp
  src/functional/network_pe.cpp
  src/engine/engine.cpp
  src/engine/network_runner.cpp
  src/energy/energy_table.cpp
  src/network/network_image.cpp
  src/stats/action.cpp
)
target_include_directories(neurort_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(neurort_core
  PUBLIC OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)
neurort_set_warnings(neurort_core)

# ---- Simulator executable ----
add_executable(neurort_sim app/main.cpp)
target_link_libraries(neurort_sim PRIVATE neurort_core)
neurort_set_warnings(neurort_sim)

# ---- Tests ----
if(NEURORT_BUILD_TESTS)
  add_subdirectory(tests)
endif()
