# Shared warning configuration. Call neurort_set_warnings(<target>) per target.
option(NEURORT_WERROR "Treat warnings as errors" OFF)

function(neurort_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow)
    if(NEURORT_WERROR)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
