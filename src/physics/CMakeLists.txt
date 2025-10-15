# Build each physics backend as its own static lib
add_subdirectory(euler)
add_subdirectory(srhd)
add_subdirectory(mhd)

# Create an interface hub that will re-export the chosen impl
add_library(aether_phys INTERFACE)

string(TOUPPER "${AETHER_PHYSICS}" AETHER_PHYSICS_UPPER)
if (AETHER_PHYSICS_UPPER STREQUAL "EULER")
  target_link_libraries(aether_phys INTERFACE aether_phys_euler)
elseif (AETHER_PHYSICS_UPPER STREQUAL "SRHD")
  target_link_libraries(aether_phys INTERFACE aether_phys_srhd)
elseif (AETHER_PHYSICS_UPPER STREQUAL "MHD")
  target_link_libraries(aether_phys INTERFACE aether_phys_mhd)
else()
  message(FATAL_ERROR "Unknown AETHER_PHYSICS='${AETHER_PHYSICS}'. Expected: Euler|SRHD|MHD")
endif()