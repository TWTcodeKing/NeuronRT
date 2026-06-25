# Fails if any functional/ or timing/ source includes an energy/ header.
# Enforces the 3-layer decoupling: energy is a pure DOWNSTREAM consumer of stats action counters
# and must never be referenced by the functional or timing layers. Invoked as a ctest.
#
# Usage: cmake -DSRC=<repo>/sim -P DecouplingAudit.cmake

file(GLOB_RECURSE files
  ${SRC}/include/neurort/functional/*.hpp
  ${SRC}/include/neurort/timing/*.hpp
  ${SRC}/src/functional/*.cpp
  ${SRC}/src/timing/*.cpp)

set(violations "")
foreach(f ${files})
  file(READ ${f} content)
  if(content MATCHES "neurort/energy/")
    list(APPEND violations ${f})
  endif()
endforeach()

if(violations)
  message(FATAL_ERROR "Decoupling violation -- functional/timing must not include energy/: ${violations}")
endif()

list(LENGTH files n)
message(STATUS "decoupling_audit OK: scanned ${n} functional/timing sources, no energy/ includes")
