#pragma once
#include <aether/core/config_build.hpp>

namespace aether::phys_ct{

inline constexpr int dim = AETHER_DIM;
    
#if AETHER_PHYSICS_KIND == 1 // Hydrodynamics
    inline constexpr int numvar = 2 + dim;
#elif AETHER_PHYSICS_KIND == 2 // SR Hydrodynamics
    inline constexpr int numvar = 5;
#elif AETHER_PHYSICS_KIND == 3 // (M)Hydrodynamics
    inline constexpr int numvar = 2 + 2*dim;
#else 
#error "Unknown AETHER_PHYSICS_KIND"
    inline constexpr int numvar = 0;
#endif
}
