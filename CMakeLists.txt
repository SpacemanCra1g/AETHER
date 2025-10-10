cmake_minimum_required(VERSION 3.20)

# ---- Project basics ----------------------------------------------------------
project(AETHER VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd/VSCode

# ---- User options  -----------------------------------------------------------
option(ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(ENABLE_MPI    "Enable MPI support"        OFF)
option(ENABLE_CUDA   "Enable CUDA support"       OFF)
option(ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" ON)

# Compile-time knobs
set(AETHER_DIM "3" CACHE STRING "Spatial dimension: 1, 2, or 3")
set_property(CACHE AETHER_DIM PROPERTY STRINGS 1 2 3)

set(AETHER_PHYSICS "Euler" CACHE STRING "Physics module: Euler, MHD, SRHD")
set_property(CACHE AETHER_PHYSICS PROPERTY STRINGS Euler MHD SRHD)

# ---- CUDA language support if enabled -----------------
if(ENABLE_CUDA)
  enable_language(CUDA)     
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ---- Configure compile-time config header ------------------------------------
string(TOUPPER "${AETHER_PHYSICS}" AETHER_PHYSICS_UPPER)
set(AETHER_PHYSICS_EULER 0)
set(AETHER_PHYSICS_MHD   0)
set(AETHER_PHYSICS_SRHD  0)
if(AETHER_PHYSICS_UPPER STREQUAL "EULER")
  set(AETHER_PHYSICS_EULER 1)
elseif(AETHER_PHYSICS_UPPER STREQUAL "MHD")
  set(AETHER_PHYSICS_MHD 1)
elseif(AETHER_PHYSICS_UPPER STREQUAL "SRHD")
  set(AETHER_PHYSICS_SRHD 1)
endif()

# Mirror feature toggles for the config header (as 0/1)
set(AETHER_ENABLE_OPENMP ${ENABLE_OPENMP})
set(AETHER_ENABLE_MPI    ${ENABLE_MPI})
set(AETHER_ENABLE_CUDA   ${ENABLE_CUDA})

# Expose project version to the template
set(AETHER_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(AETHER_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(AETHER_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# Generate build-dir header:
#   source:  <src>/include/aether/config_build.hpp.in
#   output:  <build>/generated/aether/config_build.hpp
configure_file(
  ${CMAKE_SOURCE_DIR}/include/aether/config_build.hpp.in
  ${CMAKE_BINARY_DIR}/generated/aether/config_build.hpp
  @ONLY
)

# ---- Core library target -----------------------------------------------------
add_library(aether_core)

# Gather sources 
file(GLOB_RECURSE AETHER_CORE_SRC CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/*.cpp")
target_sources(aether_core PRIVATE ${AETHER_CORE_SRC})

# Public headers live in include/, plus the generated header dir
target_include_directories(aether_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
)

# Language features & warnings
target_compile_features(aether_core PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(aether_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(aether_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

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
  file(GLOB_RECURSE AETHER_CUDA_SRC CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/src/*.cu")
  target_sources(aether_core PRIVATE ${AETHER_CUDA_SRC})
endif()

# Sanitizers in Debug (Linux/Clang/GCC)
if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  target_compile_options(aether_core PRIVATE -fsanitize=address,undefined)
  target_link_options(aether_core    PRIVATE -fsanitize=address,undefined)
endif()

# ---- Main executable ---------------------------------------------------------
add_executable(aether ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(aether PRIVATE aether_core)

# ---- Tests (optional, later) -------------------------------------------------
option(AETHER_ENABLE_TESTS "Enable tests" OFF)
if(AETHER_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()