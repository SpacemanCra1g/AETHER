#pragma once
#include "aether/core/simulation.hpp"

namespace aether::core{
    
template <int dim>
void ctu_half_time_correction(Simulation &sim, Simulation::View view);
}