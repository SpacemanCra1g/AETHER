#pragma once
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>

namespace aether::core {
void Riemann_dispatch(Simulation& Sim, double gamma);
}