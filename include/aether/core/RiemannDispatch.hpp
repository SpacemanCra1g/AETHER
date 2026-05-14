#pragma once
#include "aether/core/Kokkos_loopBounds.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/sweep_dir_selection_helpers.hpp>
#include <stdexcept>

using P = aether::prim::Prim;

namespace aether::core {

// ============================================================
// One Riemann sweep
//
// V may be Simulation::View or Simulation::CTUView as long as it
// exposes:
//   fxL, fxR, fx
//   fyL, fyR, fy    (when DIM > 1)
//   fzL, fzR, fz    (when DIM > 2)
//
// Face storage convention assumed:
//   F(c, q, k, j, i)
// ============================================================

template<riemann solv, sweep_dir dir, class Sim, class V>
AETHER_INLINE void Riemann_sweep(Sim& sim, V& v) noexcept {
    auto FL   = flux_left_view<dir>(v);
    auto FR   = flux_right_view<dir>(v);
    auto Flux = flux_view<dir>(v);
	auto source_flux = source_flux_view<dir>(v);

    const double gamma_P = sim.grid.gamma;
    const int quad     = sim.grid.quad;

    Kokkos::parallel_for(
        "Riemann_sweep",
        aether::loops::riemann_sweep_policy(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double gamma = gamma_P;
            for (int q = 0; q < quad; ++q) {
                aether::phys::prims L{};
                aether::phys::prims R{};
                aether::phys::prims F{};

                // Gather left state in solver-normal ordering
                L.rho = FL(P::RHO, q, k, j, i);
                L.vx  = FL(VelMap<dir>::VN,  q, k, j, i);
                L.vy  = 0.0;
                L.vz  = 0.0;
                if constexpr (P::HAS_VY) {
                    L.vy = FL(VelMap<dir>::VT1, q, k, j, i);
                }
                if constexpr (P::HAS_VZ) {
                    L.vz = FL(VelMap<dir>::VT2, q, k, j, i);
                }
                L.p   = FL(P::P, q, k, j, i);
				L.e   = FL(P::EINT, q, k, j, i);

                // Gather right state in solver-normal ordering
                R.rho = FR(P::RHO, q, k, j, i);
                R.vx  = FR(VelMap<dir>::VN,  q, k, j, i);
                R.vy  = 0.0;
                R.vz  = 0.0;
                if constexpr (P::HAS_VY) {
                    R.vy = FR(VelMap<dir>::VT1, q, k, j, i);
                }
                if constexpr (P::HAS_VZ) {
                    R.vz = FR(VelMap<dir>::VT2, q, k, j, i);
                }
                R.p   = FR(P::P, q, k, j, i);
				R.e   = FR(P::EINT, q, k, j, i);

                if constexpr (solv == riemann::hll) {
                    F = hll(L, R, gamma);
                } else if constexpr (solv == riemann::hllc) {
                    F = hllc(L, R, gamma);
                }

                // Store flux back in directional component ordering
                Flux(P::RHO, q, k, j, i) = F.rho;
                Flux(VelMap<dir>::VN,  q, k, j, i) = F.vx;
                if constexpr (P::HAS_VY) {
                    Flux(VelMap<dir>::VT1, q, k, j, i) = F.vy;
                }
                if constexpr (P::HAS_VZ) {
                    Flux(VelMap<dir>::VT2, q, k, j, i) = F.vz;
                }
                Flux(P::P, q, k, j, i) = F.p;

				// This component now contains F1 from equation (6,7) of Li 2008 paper
                Flux(P::EINT,q,k,j,i) = F.e;

				// For now the Eint term is the only source term being tracked, so this array has dim (1,q,NZ,NY,NX)
				// Remember to update this for additional sources
                source_flux(0,q,k,j,i) = F.rho * ((F.rho >= 0.0) ? 1.0 / L.rho : 1.0 / R.rho);
            }
        }
    );
}

// ============================================================
// Dispatcher
// ============================================================

template<class Sim, class V>
void Riemann_dispatch(Sim& sim, V& v) {
    auto view = sim.view();
    auto prim = view.prim;

    switch (sim.cfg.riem) {
        case riemann::hll:
            Riemann_sweep<riemann::hll, sweep_dir::x>(sim, v);
            if constexpr (Sim::dim > 1) {
                Riemann_sweep<riemann::hll, sweep_dir::y>(sim, v);
            }
            if constexpr (Sim::dim > 2) {
                Riemann_sweep<riemann::hll, sweep_dir::z>(sim, v);
            }
            break;
        case riemann::hllc:
            Riemann_sweep<riemann::hllc, sweep_dir::x>(sim, v);
            if constexpr (Sim::dim > 1) {
                Riemann_sweep<riemann::hllc, sweep_dir::y>(sim, v);
            }
            if constexpr (Sim::dim > 2) {
                Riemann_sweep<riemann::hllc, sweep_dir::z>(sim, v);
            }
            break;
        default:
            throw std::runtime_error("Riemann_dispatch: unknown Riemann solver");
    }
}

} // namespace aether::core
