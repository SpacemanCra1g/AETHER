#pragma once
#include "aether/core/views.hpp"
#include <aether/core/simulation.hpp>
#include <aether/core/config.hpp>
#include <aether/physics/counts.hpp>

namespace aether::core {
void boundary_conditions(Simulation&,CellsViewT<4>&);
}