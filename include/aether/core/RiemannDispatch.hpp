#pragma once

#include <Kokkos_Core.hpp>

#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/api.hpp>
#include <stdexcept>

using P = aether::prim::Prim;

namespace aether::core {

// ============================================================
// Velocity remapping by sweep direction
// hll() expects normal velocity in vx, transverse in vy/vz
// ============================================================

template<sweep_dir dir>
struct VelMap;

template<>
struct VelMap<sweep_dir::x> {
    static constexpr int VN  = P::VX;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VZ;
};

template<>
struct VelMap<sweep_dir::y> {
    static constexpr int VN  = P::VY;
    static constexpr int VT1 = P::VX;
    static constexpr int VT2 = P::VZ;
};

template<>
struct VelMap<sweep_dir::z> {
    static constexpr int VN  = P::VZ;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VX;
};

// ============================================================
// Face loop bounds
// Rank-3 always: (k,j,i)
// ============================================================
template<sweep_dir dir, class Sim>
auto face_halo1(const Sim& sim) {
    using exec_space = typename Sim::policy_type::execution_space;

    if constexpr (dir == sweep_dir::x) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? 1 : 0), (Sim::dim > 1 ? 1 : 0), 1 },
            { sim.xfaces.Nz - (Sim::dim > 2 ? 1 : 0),
              sim.xfaces.Ny - (Sim::dim > 1 ? 1 : 0),
              sim.xfaces.Nfx - 1 }
        );
    }
    else if constexpr (dir == sweep_dir::y) {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { (Sim::dim > 2 ? 1 : 0), 1, (Sim::dim > 1 ? 1 : 0) },
            { sim.yfaces.Nz - (Sim::dim > 2 ? 1 : 0),
              sim.yfaces.Nfy - 1,
              sim.yfaces.Nx - 1 }
        );
    }
    else {
        return Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            { 1, 1, 1 },
            { sim.zfaces.Nfz - 1,
              sim.zfaces.Ny  - 1,
              sim.zfaces.Nx  - 1 }
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

// ============================================================
// Select the face-state buffers for a sweep
// ============================================================

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_left_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fxL;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fyL;
    }
    else {
        return v.fzL;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_right_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fxR;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fyR;
    }
    else {
        return v.fzR;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fx;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fy;
    }
    else {
        return v.fz;
    }
}

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

    const double gamma_P = sim.grid.gamma;
    const int quad     = sim.grid.quad;

    Kokkos::parallel_for(
        "Riemann_sweep",
        face_halo1<dir>(sim),
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

                if constexpr (solv == riemann::hll) {
                    F = hll(L, R, gamma);
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

        default:
            throw std::runtime_error("Riemann_dispatch: unknown Riemann solver");
    }
}

} // namespace aether::core