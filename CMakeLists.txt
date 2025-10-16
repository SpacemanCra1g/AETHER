cmake_minimum_required(VERSION 3.20)

# ---- Project basics ----------------------------------------------------------
project(AETHER VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd/VSCode

# ---- User options  -----------------------------------------------------------
option(ENABLE_OPENMP "Enable OpenMP parallelism" OFF)
option(ENABLE_MPI    "Enable MPI support"        OFF)
option(ENABLE_CUDA   "Enable CUDA support"       OFF)
option(ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" ON)
option(AETHER_BOUNDS_CHECK "Enable runtime bounds checking" OFF)

# Compile-time knobs
set(AETHER_DIM "1" CACHE STRING "Spatial dimension: 1, 2, or 3")
set_property(CACHE AETHER_DIM PROPERTY STRINGS 1 2 3)

set(AETHER_PHYSICS "Euler" CACHE STRING "Physics module: Euler, MHD, SRHD")
set_property(CACHE AETHER_PHYSICS PROPERTY STRINGS Euler MHD SRHD)

# ---- CUDA language support if enabled -----------------
if(ENABLE_CUDA)
  enable_language(CUDA)     
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Mirror feature toggles for the config header (as 0/1)
set(AETHER_ENABLE_OPENMP ${ENABLE_OPENMP})
set(AETHER_ENABLE_MPI    ${ENABLE_MPI})
set(AETHER_ENABLE_CUDA   ${ENABLE_CUDA})
# set(AETHER_BOUNDS_CHECK  ${AETHER_BOUNDS_CHECK})

# Expose project version to the template
set(AETHER_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(AETHER_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(AETHER_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# ---- Map options to 0/1 ints for headers ----
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

# Map physics name -> small integer (for cheap #if)
set(AETHER_PHYSICS_KIND 0)
if(AETHER_PHYSICS STREQUAL "Euler")
  set(AETHER_PHYSICS_KIND 1)
elseif(AETHER_PHYSICS STREQUAL "SRHD")
  set(AETHER_PHYSICS_KIND 2)
elseif(AETHER_PHYSICS STREQUAL "MHD")
  set(AETHER_PHYSICS_KIND 3)
else()
  message(FATAL_ERROR "Unknown AETHER_PHYSICS='${AETHER_PHYSICS}'")
endif()

# Generate build-dir header:
#   source:  <proj_src>/include/aether/core/config_build.hpp.in
#   output:  <build>/generated/aether/core/config_build.hpp
configure_file(
  ${CMAKE_SOURCE_DIR}/include/aether/core/config_build.hpp.in.cmake
  ${CMAKE_BINARY_DIR}/generated/aether/core/config_build.hpp
  @ONLY
)
# ---- Shared compile options (DRY across core + physics) ---------------------
add_library(aether_compile_options INTERFACE)
if(MSVC)
  target_compile_options(aether_compile_options INTERFACE /W4 /permissive-)
else()
  target_compile_options(aether_compile_options INTERFACE -Wall -Wextra -Wpedantic)
endif()

# ---- Core library target -----------------------------------------------------
add_library(aether_core)

# Gather sources 
file(GLOB_RECURSE AETHER_CORE_CPP CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/*.cpp")
     
# Exclude the physics/ subdirctory (property of the physics libs)     
list(FILTER AETHER_CORE_CPP EXCLUDE REGEX ".*/src[\\/]physics/.*")

target_sources(aether_core PRIVATE ${AETHER_CORE_CPP})

# Public headers live in include/, plus the generated header dir
target_include_directories(aether_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
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

# CUDA (wires in .cu files if necessary)
if(ENABLE_CUDA)
  file(GLOB_RECURSE AETHER_CORE_CU CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/src/*.cu")
  list(FILTER AETHER_CORE_CU EXCLUDE REGEX ".*/src[\\/]physics/.*")
  target_sources(aether_core PRIVATE ${AETHER_CORE_CU})
endif()

# Sanitizers in Debug (Linux/Clang/GCC)
if(ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(aether_core PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  target_link_options(aether_core PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

# Bounds-check define propagated to dependents 
if (AETHER_BOUNDS_CHECK)
  target_compile_definitions(aether_core PUBLIC AETHER_BOUNDS_CHECK=1)
endif()

# Links Physics libraries
add_subdirectory(${CMAKE_SOURCE_DIR}/src/physics)
target_link_libraries(aether_core PUBLIC aether_phys)

# ---- Main executable ---------------------------------------------------------
add_executable(aether.ex ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(aether.ex PRIVATE aether_core aether_phys)

# ---- Tests (optional, later) -------------------------------------------------
option(AETHER_ENABLE_TESTS "Enable tests" OFF)
if(AETHER_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT AETHER_IPO_OK OUTPUT AETHER_IPO_MSG)
if (AETHER_IPO_OK)
  foreach(tgt aether_core aether_phys_euler aether_phys_srhd aether_phys_mhd)
    set_property(TARGET ${tgt} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET ${tgt} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
  endforeach()
else()
  message(STATUS "IPO/LTO not supported: ${AETHER_IPO_MSG}")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<CONFIG:Release>:-O3 -fstrict-aliasing>)
endif()

add_custom_target(aether_copy_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
  COMMENT "Updating compile_commands.json in source root for clangd"
)