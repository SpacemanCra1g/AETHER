#include "aether/core/enums.hpp"
#include "aether/physics/euler/convert.hpp"
#include <aether/core/RiemannDispatch.hpp>
#include <aether/core/SpaceDispatch.hpp>
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/core/flux_difference.hpp>
#include <aether/core/TemporalDispatch.hpp>
#include <iostream>

namespace aether::core{

void Time_stepper(Simulation& sim){
    auto domain = sim.view();
    switch (sim.cfg.time_step) {

        case time_stepper::rk1:
            Space_solve(sim);
            if (sim.ctu_enabled) CTU_correction(sim);
            Riemann_dispatch(sim, domain);
            flux_diff_sweep(domain.prim, sim);
            axpy(domain.cons, -1.0, domain.prim);
            
            break;
        
        case time_stepper::char_trace:
            Space_solve(sim);
            if (sim.ctu_enabled) CTU_correction(sim);
            Riemann_dispatch(sim, domain);
            flux_diff_sweep(domain.prim, sim);
            axpy(domain.cons, -1.0, domain.prim);
            break;

        default:
            throw std::runtime_error("Time_stepper: unknown time stepper");
    }
}

} // namespace aether::core