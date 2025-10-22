#pragma once
#include "aether/core/views.hpp"
#include <aether/core/simulation.hpp>
#include <aether/core/config.hpp>

namespace aether::core {
AETHER_INLINE void boundary_conditions(Simulation&,CellsViewT<AETHER_DIM>&);
}