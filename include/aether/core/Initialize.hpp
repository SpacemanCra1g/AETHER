#pragma once
#include <aether/core/simulation.hpp>

namespace aether::core {

template<typename Sim>
void initialize_domain(Sim&);

extern template void initialize_domain<Simulation>(Simulation &);
}
