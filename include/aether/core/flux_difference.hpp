#pragma once

#include <Kokkos_Core.hpp>

#include <aether/physics/counts.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>

namespace loop = aether::loops;
namespace aether::core {

template <int numvar, sweep_dir dir, class Sim, class CellViewT, class FaceViewT>
AETHER_INLINE
static void flux_sweep(CellViewT out, FaceViewT FW, Sim& sim) {
    const double dtdx_p =
        (dir == sweep_dir::x) ? (sim.time.dt / sim.grid.dx) :
        (dir == sweep_dir::y) ? (sim.time.dt / sim.grid.dy) :
                                (sim.time.dt / sim.grid.dz);

    const int ioff = (dir == sweep_dir::x) ? 1 : 0;
    const int joff = (dir == sweep_dir::y) ? 1 : 0;
    const int koff = (dir == sweep_dir::z) ? 1 : 0;

    Kokkos::parallel_for(
        "flux_sweep",
        loop::cells_interior(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int c = 0; c < numvar; ++c) {
                const double FR = FW(c, 0, k + koff, j + joff, i + ioff);
                const double FL = FW(c, 0, k, j, i);
                auto out_p = out;
                const double dtdx = dtdx_p;

                if constexpr (dir == sweep_dir::x) {
                    out_p(c, k, j, i) = -dtdx * (FL - FR);
                } else {
                    out_p(c, k, j, i) += -dtdx * (FL - FR);
                }
            }
        }
    );
}

template<class Sim>
void flux_diff_sweep(CellView out, Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;

    auto view = sim.view();

    flux_sweep<numvar, sweep_dir::x>(out, view.fx, sim);

    if constexpr (AETHER_DIM > 1) {
        flux_sweep<numvar, sweep_dir::y>(out, view.fy, sim);
    }

    if constexpr (AETHER_DIM > 2) {
        flux_sweep<numvar, sweep_dir::z>(out, view.fz, sim);
    }
}

} // namespace aether::core