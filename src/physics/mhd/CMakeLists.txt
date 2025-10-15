# MHD backend
add_library(aether_phys_mhd STATIC
  active_impl.cpp
  # add more MHD-specific .cpp here
)

target_link_libraries(aether_phys_mhd PRIVATE aether_compile_options)

target_include_directories(aether_phys_mhd
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/generated
)

if(ENABLE_CUDA)
  file(GLOB_RECURSE AETHER_MHD_CU CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
  if(AETHER_MHD_CU)
    target_sources(aether_phys_mhd PRIVATE ${AETHER_MHD_CU})
  endif()
endif()