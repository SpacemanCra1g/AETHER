#pragma once
#include <aether/core/config.hpp>

namespace aether::physics::euler{
    struct prims{
        double rho{0.0}, p{0.0}, vx{0.0};
        #if AETHER_DIM > 1
        double vy{0.0};
        #endif
        #if AETHER_DIM > 2
        double vz{0.0};
        #endif
    };

    struct cons{
        double rho{0.0}, E{0.0}, mx{0.0};
        #if AETHER_DIM > 1
        double my{0.0};
        #endif
        #if AETHER_DIM > 2
        double mz{0.0};
        #endif
    };

}