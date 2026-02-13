#pragma once
#include <aether/core/config_build.hpp>

namespace aether::phys_ct{

inline constexpr int dim = AETHER_DIM;
    
#if AETHER_PHYSICS_KIND == 1 // Hydrodynamics
    // If Extra transverse velocities are tracked, add that in here
    #if AETHER_DIM_FORCING
        #if AETHER_DIM == 1
            #if AETHER_EXTRA_DIM == 2
                inline constexpr int numvar = 2 + dim + 1;
            #elif AETHER_EXTRA_DIM == 3
                inline constexpr int numvar = 2 + dim + 2;
            #endif
        #elif AETHER_DIM == 2
            #if AETHER_EXTRA_DIM == 2
                inline constexpr int numvar = 2 + dim;
            #elif AETHER_EXTRA_DIM == 3
                inline constexpr int numvar = 2 + dim + 1;
            #endif
        #endif
    #else 
        inline constexpr int numvar = 2 + dim;
    #endif
#elif AETHER_PHYSICS_KIND == 2 // SR Hydrodynamics
    inline constexpr int numvar = 5;
#elif AETHER_PHYSICS_KIND == 3 // (M)Hydrodynamics
    inline constexpr int numvar = 2 + 2*dim;
#else 
#error "Unknown AETHER_PHYSICS_KIND"
    inline constexpr int numvar = 0;
#endif
}
