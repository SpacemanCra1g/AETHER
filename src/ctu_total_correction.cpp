#include "aether/core/simulation.hpp"
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/core/RiemannDispatch.hpp>

namespace aether::core{
void CTU_correction(Simulation &sim){

    auto view = sim.view();
    Riemann_dispatch(sim, view);
    auto ctu_view = sim.ctu_buff.view();
    Riemann_dispatch(sim, ctu_view);
}

}