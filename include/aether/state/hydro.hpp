#pragma once 
#include <aether/core/config.hpp>
#include <aether/math/vec3.hpp>

namespace aether::state::hydro{

    struct Primative{
        double rho{0.0}, p{0.0};
        aether::math::Vec3 v; 
    };

    struct Conservative{
        double mass{0.0}, E{0.0};
        aether::math::Vec3 mom; 
    };

    struct Flux{
        double mass{0.0}, E{0.0};
        aether::math::Vec3 mom; 
    };

}
