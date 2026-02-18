#pragma once
#include "aether/physics/counts.hpp"
#include <aether/core/simulation.hpp>
#include <aether/core/views.hpp>
#include <aether/core/enums.hpp>
#include <cstddef>

namespace aether::core {

template <int numvar, sweep_dir dir, class Flux, class Flux_ext>
AETHER_INLINE static void flux_sweep(CellsView &out, Flux &FW, Flux_ext &F_ext, Simulation &sim){
    int il = 0;
    int jl = 0;
    int kl = 0;
    double dxt = 0.0;
    if constexpr (dir == sweep_dir::x) {
        dxt = sim.time.dt/sim.grid.dx;
        il = 1;
    }else if constexpr (dir == sweep_dir::y) {
        dxt = sim.time.dt/sim.grid.dy;
        jl = 1;
    }else{
        dxt = sim.time.dt/sim.grid.dz;
        kl = 1;
    }

    int kn = sim.grid.nz;
    int jn = sim.grid.ny;
    int in = sim.grid.nx;
    auto view = sim.view();

    // The implicit barrier is kept here, need to avoid race condition
    // on the threads that begin the 2/3D sweeps early
    #pragma omp for collapse(3) schedule(static)
    for (int k = 0; k < kn; ++k){
    for (int j = 0; j < jn; ++j){
    for (int i = 0; i < in; ++i){
        std::size_t Flux_R = F_ext.index(i + il, j + jl, k + kl);
        std::size_t Flux_L = F_ext.index(i, j, k);
        std::size_t cell = view.cons.ext.index(i,j,k);
        for (int var = 0; var < numvar; ++ var){
            if constexpr (dir == sweep_dir::x) {
                out.comp[var][cell] = dxt*(FW.comp[var][Flux_L] - FW.comp[var][Flux_R]);
            } else{
                out.comp[var][cell] += dxt*(FW.comp[var][Flux_L] - FW.comp[var][Flux_R]);
            }
        }
    }}}
}

AETHER_INLINE void flux_diff_sweep(CellsView &out, Simulation &sim) noexcept{
    constexpr int numvar = aether::phys_ct::numvar; 
    
    auto view = sim.view();
    #pragma omp parallel shared(out,sim,view)
    {   
        auto &F_ext_x = sim.flux_x_ext;
        auto &Flux_X = view.x_flux;
        flux_sweep<numvar, sweep_dir::x>(out, Flux_X, F_ext_x, sim);
        #if AETHER_DIM > 1
            auto &F_ext_y = sim.flux_y_ext;
            auto &Flux_Y = view.y_flux;
            flux_sweep<numvar, sweep_dir::y>(out, Flux_Y, F_ext_y, sim);
        #endif
        #if AETHER_DIM > 2
            auto &F_ext_z = sim.flux_z_ext;
            auto &Flux_Z = view.z_flux;
            flux_sweep<numvar, sweep_dir::z>(out, Flux_Z, F_ext_z, sim);
        #endif

    }
}
}