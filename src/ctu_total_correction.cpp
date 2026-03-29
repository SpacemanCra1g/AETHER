#include "aether/core/simulation.hpp"
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/core/CTU/ctu_half_time_step.hpp>
#include <aether/core/RiemannDispatch.hpp>

namespace aether::core{

void CTU_correction([[maybe_unused]] Simulation& sim) {

#if AETHER_DIM == 2
    auto view = sim.view();
    Riemann_dispatch(sim, view);
    ctu_half_time_correction<2>(sim, view);
#elif AETHER_DIM == 3
    auto view = sim.view();
    auto ctu  = sim.ctu_view();

    Riemann_dispatch(sim, view);
    ctu_half_time_correction<3>(sim, view);
    Riemann_dispatch(sim, ctu);
    ctu_total_correction<3>(sim, view);
#endif

}
}