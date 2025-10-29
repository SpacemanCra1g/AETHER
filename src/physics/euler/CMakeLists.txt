# Euler backend
add_library(aether_phys_euler STATIC
  active_impl.cpp
  convert.cpp
  # add more Euler-specific .cpp here
)

# Share warnings/flags and headers
target_link_libraries(aether_phys_euler PRIVATE aether_compile_options)

# Public headers live at <repo>/include so consumers see aether/...
target_include_directories(aether_phys_euler
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/generated
)

# Optional: CUDA sources in this backend (only if you add any .cu files)
if(ENABLE_CUDA)
  file(GLOB_RECURSE AETHER_EULER_CU CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
  if(AETHER_EULER_CU)
    target_sources(aether_phys_euler PRIVATE ${AETHER_EULER_CU})
  endif()
endif()