// AETHER Global configuration header
// Contains project wide macros

#pragma once

#if __has_include(<aether/config_build.hpp>)
  #include <aether/config_build.hpp>       
#endif

// ---------- Compiler/Platform feature probes ----------
#if defined(_MSC_VER)
  #define AETHER_COMPILER_MSVC 1
#else
  #define AETHER_COMPILER_MSVC 0
#endif

#if defined(__clang__)
  #define AETHER_COMPILER_CLANG 1
#else
  #define AETHER_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !AETHER_COMPILER_CLANG
  #define AETHER_COMPILER_GCC 1
#else
  #define AETHER_COMPILER_GCC 0
#endif

// ---------- CUDA host/device decoration ----------
#if defined(__CUDACC__)
  #define AETHER_HD __host__ __device__
#else
  #define AETHER_HD
#endif

// ---------- Force-inline / always-inline hint ----------
#if AETHER_COMPILER_MSVC
  #define AETHER_INLINE __forceinline
#elif AETHER_COMPILER_GCC || AETHER_COMPILER_CLANG
  #define AETHER_INLINE inline __attribute__((always_inline))
#else
  #define AETHER_INLINE inline
#endif

// ---------- Branch prediction hints ----------
#if AETHER_COMPILER_GCC || AETHER_COMPILER_CLANG
  #define AETHER_LIKELY(x)   (__builtin_expect(!!(x), 1))
  #define AETHER_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
  #define AETHER_LIKELY(x)   (x)
  #define AETHER_UNLIKELY(x) (x)
#endif

// ---------- Alignment helper ----------
#if AETHER_COMPILER_MSVC
  #define AETHER_ALIGN(N) __declspec(align(N))
#else
  #define AETHER_ALIGN(N) __attribute__((aligned(N)))
#endif

// ---------- Sanity defaults if build config wasn't included ----------
#ifndef AETHER_DIM
  #define AETHER_DIM 3          // default to 3D; overridden by generated header if present
#endif

// Optional physics flags as no-ops if not defined by the generated header
#ifndef AETHER_PHYSICS_EULER
  #define AETHER_PHYSICS_EULER 0
#endif
#ifndef AETHER_PHYSICS_MHD
  #define AETHER_PHYSICS_MHD 0
#endif
#ifndef AETHER_PHYSICS_SRHD
  #define AETHER_PHYSICS_SRHD 0
#endif

// ---------- Debug mode boundary checking ----------
#ifndef AETHER_BOUNDS_CHECK
  #if !defined (NDEBUG)
    #define AETHER_BOUNDS_CHECK 1
  #else 
    #define AETHER_BOUNDS_CHECK 0
  #endif
#endif

// ---------- Basic validation ----------
static_assert(AETHER_DIM == 1 || AETHER_DIM == 2 || AETHER_DIM == 3,
              "AETHER_DIM must be 1, 2, or 3");