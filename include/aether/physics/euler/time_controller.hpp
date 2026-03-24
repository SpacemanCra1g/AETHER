#pragma once
#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>
#include <cmath>

using P = aether::prim::Prim;
namespace loop = aether::loops;

namespace aether::physics::euler{

AETHER_INLINE 
double max_propagation_speed(aether::core::Simulation& sim){
    auto domain = sim.view();
    auto prims  = domain.prim;

    const double gamma  = domain.gamma;
    const double dx_inv = 1.0 / sim.grid.dx;
    const double dy_inv = (AETHER_DIM > 1) ? 1.0 / sim.grid.dy : 0.0;
    const double dz_inv = (AETHER_DIM > 2) ? 1.0 / sim.grid.dz : 0.0;
    double max_num = 0.0;

    Kokkos::parallel_reduce(
        "Max propagation speed",
        loop::cells_full(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i, double& local_max)
        {
            const double rho = prims(P::RHO, k, j, i);
            const double p   = prims(P::P,   k, j, i);
            const double cs  = sqrt(gamma * p / rho);
            double u_max = (fabs(prims(P::VX, k, j, i)) + cs) * dx_inv;

            if constexpr (P::HAS_VY) {
                const double v_max = fabs(prims(P::VY, k, j, i)) + cs;
                u_max = fmax(u_max, v_max * dy_inv);
            }
            if constexpr (P::HAS_VZ) {
                const double w_max = fabs(prims(P::VZ, k, j, i)) + cs;
                u_max = fmax(u_max, w_max * dz_inv);
            }
            local_max = fmax(local_max, u_max);
        },
        Kokkos::Max<double>(max_num)
    );
    return max_num;
}

AETHER_INLINE void set_dt(aether::core::Simulation& sim){
    const double lam = max_propagation_speed(sim);
    const double dt = sim.time.cfl / lam;
    if (sim.time.t + dt >= sim.time.t_end){
        sim.time.dt = sim.time.t_end-sim.time.t;
    } else{
        sim.time.dt = dt;
    }
    sim.time.t += sim.time.dt;
}
}