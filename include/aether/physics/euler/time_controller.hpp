#pragma once
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <cmath>
#include <cstddef>
#include <limits>

using P = aether::prim::Prim;
namespace aether::physics::euler{
AETHER_INLINE double max_propagation_speed(aether::core::Simulation& sim){
    const std::size_t N = sim.ext.flat();
    auto view = sim.view();
    auto  &prims = view.prim;
    double max_num = std::numeric_limits<double>::lowest();
    double cs, S_speed, u_max;
    const double gamma = sim.grid.gamma;
    const double dx_inv = 1.0/sim.grid.dx;
    const double dy_inv = (AETHER_DIM > 1) ? 1.0/sim.grid.dy : 0.0; 
    const double dz_inv = (AETHER_DIM > 2) ? 1.0/sim.grid.dz : 0.0; 
    
    #pragma omp parallel for schedule(static) default(none) shared(N, prims,gamma,dx_inv, dy_inv, dz_inv) \
            private(cs,S_speed,u_max) reduction(max:max_num) 
    for (std::size_t i = 0; i < N; ++i){
        cs = std::sqrt(gamma*prims.var(P::P,i)/prims.var(P::RHO,i));
        u_max = (std::fabs(prims.var(P::VX,i)) + cs)*dx_inv;
        if constexpr (P::HAS_VY) {
            const double v_max = std::fabs(prims.var(P::VY,i)) + cs;
            u_max = std::max(u_max,v_max*dy_inv);
        }
        if constexpr (P::HAS_VZ) {
            const double w_max = std::fabs(prims.var(P::VZ,i)) + cs;
            u_max = std::max(u_max,w_max*dz_inv);
        }
        S_speed = u_max;
        max_num = std::max(S_speed,max_num);
    }

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