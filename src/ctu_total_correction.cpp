#include "aether/core/simulation.hpp"
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/core/CTU/ctu_half_time_step.hpp>
#include <aether/core/RiemannDispatch.hpp>

namespace aether::core{

void CTU_correction([[maybe_unused]] Simulation& sim) {

#if AETHER_DIM > 1 
    auto view = sim.view();
    Riemann_dispatch(sim, view);
    ctu_half_time_correction<AETHER_DIM>(sim, view);
#endif

}
}