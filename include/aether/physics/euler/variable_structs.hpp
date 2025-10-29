#pragma once
#include <aether/core/config.hpp>

namespace aether::physics::euler{
    struct prims{
        double rho{0.0}, vx{0.0}, vy{0.0}, vz{0.0}, p{0.0};
    };

    struct cons{
        double rho{0.0}, mx{0.0},my{0.0}, mz{0.0}, E{0.0};
    };

}