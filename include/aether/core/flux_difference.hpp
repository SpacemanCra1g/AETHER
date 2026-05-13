#pragma once

#include <Kokkos_Core.hpp>
#include <aether/physics/counts.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>
#include <aether/core/sweep_dir_selection_helpers.hpp>

namespace loop = aether::loops;
using P = aether::prim::Prim;
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
	auto view = sim.view();
	auto source_flux = source_flux_view<dir>(view);
	auto source = source_view<dir>(view);

    Kokkos::parallel_for(
        "flux_sweep",
        loop::cells_interior(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto out_p = out;
            const double dtdx = dtdx_p;
            for (int c = 0; c < numvar; ++c) {
                const double FR = FW(c, 0, k + koff, j + joff, i + ioff);
                const double FL = FW(c, 0, k, j, i);

                if constexpr (dir == sweep_dir::x) {
                    out_p(c, k, j, i) = -dtdx * (FL - FR);

                } else {
                    out_p(c, k, j, i) += -dtdx * (FL - FR);
                }
            }
			// Update Eint component using p_bar (stored in source array)
			// F^2 fluxes (stored in source_flux) and F^1 stored in flux array
			// for details see reference "A Simple Dual Implementation to Track Pressure" Li 2008
			const double FR = FW(P::EINT, 0, k + koff, j + joff, i + ioff);
            const double FL = FW(P::EINT, 0, k, j, i);
			const double SR = source_flux(0, 0, k + koff, j + joff, i + ioff);
            const double SL = source_flux(0, 0, k, j, i);
			const double pbar = source(0,k,j,i);
			if constexpr (dir == sweep_dir::x) {
                out_p(P::EINT, k, j, i) = -dtdx * ((FL - FR) + pbar*(SL-SR));
			} else {
				out_p(P::EINT, k, j, i) += -dtdx * ((FL - FR) + pbar*(SL-SR));
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
