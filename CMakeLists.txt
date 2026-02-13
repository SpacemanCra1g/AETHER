cmake_minimum_required(VERSION 3.20)

# ---- Project basics ----------------------------------------------------------
project(AETHER VERSION 0.1.0 LANGUAGES CXX)

# Create a symlink once so clangd can find compile_commands.json in the build dir
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_BINARY_DIR}/compile_commands.json"
          "${CMAKE_SOURCE_DIR}/compile_commands.json"
  RESULT_VARIABLE _link_result
  OUTPUT_QUIET ERROR_QUIET
)
if(_link_result)
  message(STATUS "Could not create symlink for compile_commands.json (likely Windows); configure clangd to use the build dir instead.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd/VSCode

# ---- User options  -----------------------------------------------------------
option(ENABLE_OPENMP         "Enable OpenMP parallelism"                 OFF)
option(ENABLE_MPI            "Enable MPI support"                        OFF)
option(ENABLE_CUDA           "Enable CUDA support"                       OFF)
option(ENABLE_SANITIZERS     "Enable ASan/UBSan in Debug builds"         ON)
option(AETHER_BOUNDS_CHECK   "Enable runtime bounds checking"            OFF)

# Compile-time knobs
set(AETHER_DIM "1" CACHE STRING "Spatial dimension: 1, 2, or 3")
set_property(CACHE AETHER_DIM PROPERTY STRINGS 1 2 3)

set(AETHER_PHYSICS "Euler" CACHE STRING "Physics module: Euler, MHD, SRHD")
set_property(CACHE AETHER_PHYSICS PROPERTY STRINGS Euler MHD SRHD)

set(AETHER_FORCE_DIM "0" CACHE STRING "Set more tangent velo components than the dimension default, 0 (off), 2, 3")
set_property(CACHE AETHER_FORCE_DIM PROPERTY STRINGS 0 2 3)

# ---- CUDA language support if enabled ---------------------------------------
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ---- Map options to 0/1 ints for headers ------------------------------------

# ---- Default build type ------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

  # Allow IDEs like VSCode/CMake Tools to still offer dropdown choices
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Backend features
set(AETHER_ENABLE_OPENMP_INT 0)
if(ENABLE_OPENMP)
  set(AETHER_ENABLE_OPENMP_INT 1)
endif()

set(AETHER_ENABLE_MPI_INT 0)
if(ENABLE_MPI)
  set(AETHER_ENABLE_MPI_INT 1)
endif()

set(AETHER_ENABLE_CUDA_INT 0)
if(ENABLE_CUDA)
  set(AETHER_ENABLE_CUDA_INT 1)
endif()

set(AETHER_FORCED_DIM_INT 0)
if(AETHER_FORCE_DIM)
  set(AETHER_FORCED_DIM_INT 1)
endif()
if(AETHER_FORCE_DIM STREQUAL 1)
  set(AETHER_FORCED_DIM_INT 0)
  set(AETHER_FORCE_DIM "0")  
endif()

# Physics selection
set(AETHER_PHYSICS_KIND 0)
set(AETHER_PHYSICS_EULER_INT 0)
set(AETHER_PHYSICS_MHD_INT   0)
set(AETHER_PHYSICS_SRHD_INT  0)

if(AETHER_PHYSICS STREQUAL "Euler")
  set(AETHER_PHYSICS_KIND 1)
  set(AETHER_PHYSICS_EULER_INT 1)
elseif(AETHER_PHYSICS STREQUAL "SRHD")
  set(AETHER_PHYSICS_KIND 2)
  set(AETHER_PHYSICS_SRHD_INT 1)
elseif(AETHER_PHYSICS STREQUAL "MHD")
  set(AETHER_PHYSICS_KIND 3)
  set(AETHER_PHYSICS_MHD_INT 1)
else()
  message(FATAL_ERROR "Unknown AETHER_PHYSICS='${AETHER_PHYSICS}'")
endif()

# Bounds check (single source of truth: generated header consumes this)
if(AETHER_BOUNDS_CHECK)
  set(AETHER_BOUNDS_CHECK_INT 1)
else()
  set(AETHER_BOUNDS_CHECK_INT 0)
endif()

# Project version to the template
set(AETHER_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(AETHER_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(AETHER_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# ---- Generate build-dir header ----------------------------------------------
#   source:  <proj_src>/include/aether/core/config_build.hpp.in.cmake
#   output:  <build>/generated/aether/core/config_build.hpp
configure_file(
  ${CMAKE_SOURCE_DIR}/include/aether/core/config_build.hpp.in.cmake
  ${CMAKE_BINARY_DIR}/generated/aether/core/config_build.hpp
  @ONLY
)

# ---- Shared compile options (DRY across core + physics) ---------------------
add_library(aether_compile_options INTERFACE)

# Silence "unknown pragma" only when OpenMP is OFF
if(NOT ENABLE_OPENMP)
  target_compile_options(aether_compile_options INTERFACE
    # GCC / Clang / AppleClang
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-unknown-pragmas>
    # MSVC (and clang-cl when reported as MSVC)
    $<$<CXX_COMPILER_ID:MSVC>:/wd4068>
  )
endif()

if(MSVC)
  target_compile_options(aether_compile_options INTERFACE /W4 /permissive-)
else()
  target_compile_options(aether_compile_options INTERFACE -Wall -Wextra -Wpedantic)
endif()

# ---- Sanitizers helper (list-based; INTERFACE-safe) --------------------------
set(AETHER_SAN_LIST address undefined)  # list of sanitizers

function(aether_enable_sanitizers_for TARGET_NAME)
  if(NOT ENABLE_SANITIZERS OR MSVC)
    return()
  endif()
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    return()
  endif()

  get_target_property(_type ${TARGET_NAME} TYPE)
  if(NOT _type)
    message(FATAL_ERROR "Target '${TARGET_NAME}' not found for sanitizer enabling.")
  endif()

  set(_compile_flags -fno-omit-frame-pointer)
  set(_link_flags)
  foreach(san IN LISTS AETHER_SAN_LIST)
    list(APPEND _compile_flags "-fsanitize=${san}")
    list(APPEND _link_flags    "-fsanitize=${san}")
  endforeach()

  if(_type STREQUAL "INTERFACE_LIBRARY")
    target_compile_options(${TARGET_NAME} INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${_compile_flags}>
    )
    target_link_options(${TARGET_NAME} INTERFACE ${_link_flags})
  else()
    target_compile_options(${TARGET_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${_compile_flags}>
    )
    target_link_options(${TARGET_NAME} PRIVATE ${_link_flags})
  endif()
endfunction()

# ---- Core library target -----------------------------------------------------
add_library(aether_core)

# Gather sources (no CONFIGURE_DEPENDS to avoid constant reconfigures)
file(GLOB_RECURSE AETHER_CORE_CPP
     "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Exclude physics subdir and editor/build dirs
list(FILTER AETHER_CORE_CPP EXCLUDE REGEX ".*/src[\\/]physics/.*")
list(FILTER AETHER_CORE_CPP EXCLUDE REGEX ".*/(\\.git|\\.vscode|\\.idea|build)/.*")

target_sources(aether_core PRIVATE ${AETHER_CORE_CPP})

# Public headers (generated marked as SYSTEM to reduce warning noise)
target_include_directories(aether_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  SYSTEM PUBLIC
    ${CMAKE_BINARY_DIR}/generated
)

# Apply shared compile options
target_link_libraries(aether_core PRIVATE aether_compile_options)

# OpenMP (CPU parallel loops)
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(aether_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# MPI (off by default)
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(aether_core PUBLIC MPI::MPI_CXX)
endif()

# CUDA sources only when enabled (no CONFIGURE_DEPENDS)
if(ENABLE_CUDA)
  file(GLOB_RECURSE AETHER_CORE_CU
       "${CMAKE_SOURCE_DIR}/src/*.cu")
  list(FILTER AETHER_CORE_CU EXCLUDE REGEX ".*/src[\\/]physics/.*")
  list(FILTER AETHER_CORE_CU EXCLUDE REGEX ".*/(\\.git|\\.vscode|\\.idea|build)/.*")
  target_sources(aether_core PRIVATE ${AETHER_CORE_CU})
endif()

# ---- Physics libraries -------------------------------------------------------
add_subdirectory(${CMAKE_SOURCE_DIR}/src/physics)
target_link_libraries(aether_core PUBLIC aether_phys)

# ---- Main executable ---------------------------------------------------------
add_executable(aether.ex ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(aether.ex PRIVATE aether_core aether_phys)

# ---- Enable sanitizers on linkable targets ----------------------------------
# (Key: the final link of aether.ex must carry -fsanitize=... )
aether_enable_sanitizers_for(aether_core)
aether_enable_sanitizers_for(aether_phys)  # INTERFACE-safe
aether_enable_sanitizers_for(aether.ex)

# ---- Tests (optional) --------------------------------------------------------
option(AETHER_ENABLE_TESTS "Enable tests" OFF)
if(AETHER_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- IPO/LTO guarded per-existing target ------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT AETHER_IPO_OK OUTPUT AETHER_IPO_MSG)
if (AETHER_IPO_OK)
  foreach(tgt aether_core aether_phys aether_phys_euler aether_phys_srhd aether_phys_mhd)
    if(TARGET ${tgt})
      set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
      set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endif()
  endforeach()
else()
  message(STATUS "IPO/LTO not supported: ${AETHER_IPO_MSG}")
endif()

# ---- Release-only C++ flags (directory-scoped; safe for INTERFACE libs) -----
add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -fstrict-aliasing>)
