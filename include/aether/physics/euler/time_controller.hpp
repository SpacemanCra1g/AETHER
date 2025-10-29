#pragma once
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <cmath>
#include <cstddef>
#include <limits>

using P = aether::prim::Prim;
namespace aether::physics::euler{
AETHER_INLINE double max_signal_speed(aether::core::Simulation& sim){
    const std::size_t N = sim.ext.flat();
    auto view = sim.view();
    auto  &prims = view.prim;
    double max_num = std::numeric_limits<double>::lowest();
    double cs, S_speed, u_max;
    double gamma = sim.grid.gamma;

    #pragma omp parallel for schedule(static) default(none) shared(N, prims,gamma) private(cs,S_speed,u_max) reduction(max:max_num)
    for (std::size_t i = 0; i < N; ++i){
        cs = std::sqrt(gamma*prims.var(P::P,i)/prims.var(P::RHO,i));
        u_max = std::fabs(prims.var(P::VX,i));
        if constexpr (P::HAS_VY) {
            const double v_max = std::fabs(prims.var(P::VY,i));
            u_max = std::max(u_max,v_max);
        }
        if constexpr (P::HAS_VZ) {
            const double w_max = std::fabs(prims.var(P::VZ,i));
            u_max = std::max(u_max,w_max);
        }
        S_speed = u_max + cs;
        max_num = (S_speed > max_num) ? S_speed : max_num;
    }

    return max_num;
}

}