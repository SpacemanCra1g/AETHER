#include <aether/core/CTU/ctu_transverse_correction.hpp>

namespace aether::core {

void ctu_flux_correction(Simulation &sim){
    constexpr int numvar = aether::phys_ct::numvar; 
    
    auto view = sim.view();
    auto ctu_view = sim.ctu_buff.view();
    double dt = sim.time.dt;
    #pragma omp parallel shared(sim,view,ctu_view,dt)
    {   
    #if AETHER_DIM > 1
        correction_sweep_params<sweep_dir::x, sweep_dir::y> params_xy(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,true>(params_xy);

        correction_sweep_params<sweep_dir::y, sweep_dir::x> params_yx(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,true>(params_yx);
    #endif

    #if AETHER_DIM > 2
        correction_sweep_params<sweep_dir::x, sweep_dir::z> params_xz(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,false>(params_xz);

        correction_sweep_params<sweep_dir::y, sweep_dir::z> params_yz(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,false>(params_yz);

        correction_sweep_params<sweep_dir::z, sweep_dir::x> params_zx(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,true>(params_zx);

        correction_sweep_params<sweep_dir::z, sweep_dir::y> params_zy(sim,view,ctu_view,dt);
        flux_correction_sweep<numvar,false>(params_zy);
    #endif

    }
}
}