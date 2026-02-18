#include "aether/core/simulation.hpp"
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/core/CTU/ctu_transverse_correction.hpp>
#include <aether/core/CTU/half_step_flux_update.hpp>
#include <aether/core/RiemannDispatch.hpp>

namespace aether::core{
void CTU_correction(Simulation &sim){

    half_step_update(sim);
    auto ctu_view = sim.ctu_buff.view();
    Riemann_dispatch(sim, ctu_view);
    ctu_flux_correction(sim);
}

}