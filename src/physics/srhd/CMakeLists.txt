add_library(aether_phys_srhd STATIC
  active_impl.cpp
  # add more SRHD-specific .cpp here
)

target_link_libraries(aether_phys_srhd PRIVATE aether_compile_options)

target_include_directories(aether_phys_srhd
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/generated
)

if(ENABLE_CUDA)
  file(GLOB_RECURSE AETHER_SRHD_CU CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
  if(AETHER_SRHD_CU)
    target_sources(aether_phys_srhd PRIVATE ${AETHER_SRHD_CU})
  endif()
endif()