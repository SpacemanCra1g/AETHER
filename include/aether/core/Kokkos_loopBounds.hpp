#pragma once

#include <Kokkos_Core.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/enums.hpp>

using sweep_dir = aether::core::sweep_dir;
namespace aether::loops {

template<class Sim>
using exec_space_t = typename Sim::policy_type::execution_space;


template<class Sim>
auto cells_halo3(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    int i0 = sim.cells.ibegin() -3;
    int iN = sim.cells.iend() + 3;
    int j0 = sim.cells.jbegin();
    int jN = sim.cells.jend();
    int k0 = sim.cells.kbegin();
    int kN = sim.cells.kend();
    if constexpr (AETHER_DIM > 1) {
    j0 -= 3; jN += 3;
    }
    if constexpr (AETHER_DIM > 2) {
    k0 -= 3; kN += 3;
    }
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {k0, j0, i0},
        {kN, jN, iN}
    );
}


template<class Sim>
auto cells_halo2(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    int i0 = sim.cells.ibegin() -2;
    int iN = sim.cells.iend() + 2;
    int j0 = sim.cells.jbegin();
    int jN = sim.cells.jend();
    int k0 = sim.cells.kbegin();
    int kN = sim.cells.kend();
    if constexpr (AETHER_DIM > 1) {
    j0 -= 2; jN += 2;
    }
    if constexpr (AETHER_DIM > 2) {
    k0 -= 2; kN += 2;
    }
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {k0, j0, i0},
        {kN, jN, iN}
    );
}

template<class Sim>
auto cells_halo1(const Sim& sim) {
    using exec_space = exec_space_t<Sim>;
    int i0 = sim.cells.ibegin() -1;
    int iN = sim.cells.iend() + 1;
    int j0 = sim.cells.jbegin();
    int jN = sim.cells.jend();
    int k0 = sim.cells.kbegin();
    int kN = sim.cells.kend();
    if constexpr (AETHER_DIM > 1) {
    j0 -= 1; jN += 1;
    }
    if constexpr (AETHER_DIM > 2) {
    k0 -= 1; kN += 1;
    }
    return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
        {k0, j0, i0},
        {kN, jN, iN}
    );
}

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


// ============================================================
// Face loop bounds
// Rank-3 always: (k,j,i)
// ============================================================
template<sweep_dir dir, class Sim>
auto face_halo1(const Sim& sim) {
    using exec_space = typename Sim::policy_type::execution_space;

    const int ng = sim.grid.ng;

    if constexpr (dir == sweep_dir::x) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? ng : 0),
              (Sim::dim > 1 ? ng : 0),
              ng },
            { sim.xfaces.Nz  - (Sim::dim > 2 ? ng : 0),
              sim.xfaces.Ny  - (Sim::dim > 1 ? ng : 0),
              sim.xfaces.Nfx - ng }
        );
    }
    else if constexpr (dir == sweep_dir::y) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? ng : 0),
              ng,
              (Sim::dim > 1 ? ng : 0) },
            { sim.yfaces.Nz  - (Sim::dim > 2 ? ng : 0),
              sim.yfaces.Nfy - ng,
              sim.yfaces.Nx  - (Sim::dim > 1 ? ng : 0) }
        );
    }
    else {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { ng, ng, ng },
            { sim.zfaces.Nfz - ng,
              sim.zfaces.Ny  - ng,
              sim.zfaces.Nx  - ng }
        );
    }
}

template<sweep_dir dir, class Sim>
auto face_halo2(const Sim& sim) {
    using exec_space = typename Sim::policy_type::execution_space;

    if constexpr (dir == sweep_dir::x) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? 2 : 0), (Sim::dim > 1 ? 2 : 0), 2 },
            { sim.xfaces.Nz - (Sim::dim > 2 ? 2 : 0),
              sim.xfaces.Ny - (Sim::dim > 1 ? 2 : 0),
              sim.xfaces.Nfx - 2 }
        );
    }
    else if constexpr (dir == sweep_dir::y) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? 2 : 0), 2, (Sim::dim > 1 ? 2 : 0) },
            { sim.yfaces.Nz - (Sim::dim > 2 ? 2 : 0),
              sim.yfaces.Nfy - 2,
              sim.yfaces.Nx - 2 }
        );
    }
    else {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { 2, 2, 2 },
            { sim.zfaces.Nfz - 2,
              sim.zfaces.Ny  - 2,
              sim.zfaces.Nx  - 2 }
        );
    }
}


} // namespace aether::loops