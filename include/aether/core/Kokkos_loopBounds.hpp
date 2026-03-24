#pragma once

#include <Kokkos_Core.hpp>
#include <aether/core/simulation.hpp>

namespace aether::loops {

template<class Sim>
using exec_space_t = typename Sim::policy_type::execution_space;

template<class Sim>
auto cells_full(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {0, 0, 0},
        {sim.cells.Nz, sim.cells.Ny, sim.cells.Nx}
    );
}

template<class Sim>
auto cells_interior(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {sim.cells.kbegin(), sim.cells.jbegin(), sim.cells.ibegin()},
        {sim.cells.kend(),   sim.cells.jend(),   sim.cells.iend()}
    );
}

template<class Sim>
auto xfaces_full(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {0, 0, 0},
        {sim.xfaces.Nz, sim.xfaces.Ny, sim.xfaces.Nfx}
    );
}

} // namespace aether::loops