#pragma once
#include "aether/core/views.hpp"
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/simulation.hpp>

namespace aether::core {

#if AETHER_DIM > 1

template <sweep_dir normal_dir, sweep_dir transverse_dir>
struct correction_sweep_params;

// x-y corner transport 
template <> struct correction_sweep_params<sweep_dir::x,sweep_dir::y>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridX &Flux_normal_ext;
    FaceGridY &Flux_transverse_ext;
    double dxt_half;
    Simulation &sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.x_flux_right), U_L_plus_points(view.x_flux_left)
        , U_Rstar_minus(ctu_view.x_flux_right_view), U_Lstar_plus(ctu_view.x_flux_left_view)
        , transverse_fluxes(ctu_view.y_flux_view), Flux_normal_ext(sim.flux_x_ext)
        , Flux_transverse_ext(sim.flux_y_ext), dxt_half(dt*.5/sim.grid.dy)
        , sim(sim)
        {}
};

// y-x corner transport
template <> struct correction_sweep_params<sweep_dir::y,sweep_dir::x>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridY &Flux_normal_ext;
    FaceGridX &Flux_transverse_ext;
    double dxt_half;
    Simulation &sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.y_flux_right), U_L_plus_points(view.y_flux_left)
        , U_Rstar_minus(ctu_view.y_flux_right_view), U_Lstar_plus(ctu_view.y_flux_left_view)
        , transverse_fluxes(ctu_view.x_flux_view), Flux_normal_ext(sim.flux_y_ext)
        , Flux_transverse_ext(sim.flux_x_ext), dxt_half(dt*.5/sim.grid.dx)
        , sim(sim)
        {}
};
#endif

#if AETHER_DIM > 2
// x-z corner transport
template <> struct correction_sweep_params<sweep_dir::x,sweep_dir::z>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridX &Flux_normal_ext;
    FaceGridZ &Flux_transverse_ext;
    double dxt_half;
    Simulation &sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.x_flux_right), U_L_plus_points(view.x_flux_left)
        , U_Rstar_minus(ctu_view.x_flux_right_view), U_Lstar_plus(ctu_view.x_flux_left_view)
        , transverse_fluxes(ctu_view.z_flux_view), Flux_normal_ext(sim.flux_x_ext)
        , Flux_transverse_ext(sim.flux_z_ext), dxt_half(dt*.5/sim.grid.dz)
        , sim(sim)
        {}
};

// y-z corner transport
template <> struct correction_sweep_params<sweep_dir::y,sweep_dir::z>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridY &Flux_normal_ext;
    FaceGridZ &Flux_transverse_ext;
    double dxt_half;
    Simulation &sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.y_flux_right), U_L_plus_points(view.y_flux_left)
        , U_Rstar_minus(ctu_view.y_flux_right_view), U_Lstar_plus(ctu_view.y_flux_left_view)
        , transverse_fluxes(ctu_view.z_flux_view), Flux_normal_ext(sim.flux_y_ext)
        , Flux_transverse_ext(sim.flux_z_ext), dxt_half(dt*.5/sim.grid.dz)
        , sim(sim)
        {}
};

// z-x corner transport
template <> struct correction_sweep_params<sweep_dir::z,sweep_dir::x>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridZ &Flux_normal_ext;
    FaceGridX &Flux_transverse_ext;
    double dxt_half;
    Simulation &sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.z_flux_right), U_L_plus_points(view.z_flux_left)
        , U_Rstar_minus(ctu_view.z_flux_right_view), U_Lstar_plus(ctu_view.z_flux_left_view)
        , transverse_fluxes(ctu_view.x_flux_view), Flux_normal_ext(sim.flux_z_ext)
        , Flux_transverse_ext(sim.flux_x_ext), dxt_half(dt*.5/sim.grid.dx)
        , sim(sim)
        {}
};

// z-y corner transport
template <> struct correction_sweep_params<sweep_dir::z,sweep_dir::y>{
    FaceArrayView &U_R_minus_points, &U_L_plus_points, &U_Rstar_minus, &U_Lstar_plus;
    FaceArrayView &transverse_fluxes;
    FaceGridZ &Flux_normal_ext;
    FaceGridY &Flux_transverse_ext;
    double dxt_half;
    Simulation& sim;
    correction_sweep_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view, double dt)
        : U_R_minus_points(view.z_flux_right), U_L_plus_points(view.z_flux_left)
        , U_Rstar_minus(ctu_view.z_flux_right_view), U_Lstar_plus(ctu_view.z_flux_left_view)
        , transverse_fluxes(ctu_view.y_flux_view), Flux_normal_ext(sim.flux_z_ext)
        , Flux_transverse_ext(sim.flux_y_ext), dxt_half(dt*.5/sim.grid.dy)
        , sim(sim)
        {}
};


#endif

template <int numvar, bool first_pass, sweep_dir n_dir, sweep_dir T_dir>
AETHER_INLINE static void flux_correction_sweep(correction_sweep_params<n_dir,T_dir> &params){

    int il = 0, it = 0;
    int jl = 0, jt = 0;
    int kl = 0, kt = 0;
    if constexpr (n_dir == sweep_dir::x) il = 1;
    else if constexpr (n_dir == sweep_dir::y) jl = 1;
    else kl = 1;

    if constexpr (T_dir == sweep_dir::x) it = 1;
    else if constexpr (T_dir == sweep_dir::y) jt = 1;
    else kt = 1;

    int kn = params.sim.grid.nz;
    int jn = params.sim.grid.ny;
    int in = params.sim.grid.nx;

    // The implicit barrier is kept here, need to avoid race condition
    // on the threads that begin the z/y corrections early
    #pragma omp for collapse(3) schedule(static) 
    for (int k = -kl; k < kn + 2*kl; ++k)
    for (int j = -jl; j < jn + 2*jl; ++j)
    for (int i = -il; i < in + 2*il; ++i){
        std::size_t Flux_i_minus_half_idx = params.Flux_normal_ext.index(i, j, k);

        std::size_t Flux_j_plus_half_left = params.Flux_transverse_ext.index((i-il)+it, (j-jl)+jt,(k-kl)+kt);
        std::size_t Flux_j_plus_half = params.Flux_transverse_ext.index((i)+it, (j)+jt,(k)+kt);

        std::size_t Flux_j_minus_half_left = params.Flux_transverse_ext.index((i-il), (j-jl),(k-kl));
        std::size_t Flux_j_minus_half = params.Flux_transverse_ext.index(i, j, k);


        for (int var = 0; var < numvar; ++ var){
            double correction_plus_j_corner = 0.5*
                (params.transverse_fluxes.comp[var][Flux_j_plus_half_left] + params.transverse_fluxes.comp[var][Flux_j_plus_half]);

            double correction_minus_j_corner = 0.5*
                (params.transverse_fluxes.comp[var][Flux_j_minus_half_left] + params.transverse_fluxes.comp[var][Flux_j_minus_half]);

            double correction = params.dxt_half*(correction_plus_j_corner - correction_minus_j_corner);

            if constexpr (first_pass) {
                params.U_R_minus_points.comp[var][Flux_i_minus_half_idx] = params.U_Rstar_minus.comp[var][Flux_i_minus_half_idx] - correction;

                params.U_L_plus_points.comp[var][Flux_i_minus_half_idx] = params.U_Lstar_plus.comp[var][Flux_i_minus_half_idx] - correction;
            } else{
                params.U_R_minus_points.comp[var][Flux_i_minus_half_idx] -=   correction;
                params.U_L_plus_points.comp[var][Flux_i_minus_half_idx]  -=   correction;
            }
            
        }
    }
}

void ctu_flux_correction(Simulation &sim);
}