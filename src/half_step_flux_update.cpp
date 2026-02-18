#include "aether/core/views.hpp"
#include <aether/core/CTU/half_step_flux_update.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/simulation.hpp>


namespace aether::core{

template<sweep_dir dir>
struct half_step_params;


template<>
struct half_step_params<sweep_dir::x>{
    FaceGridX &flux_ext;
    FaceArrayView &flux_right, &flux_left, &ctu_flux_right, &ctu_flux_left;

    half_step_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view):
      flux_ext(sim.flux_x_ext)
    , flux_right(view.x_flux_right)
    , flux_left(view.x_flux_left)
    , ctu_flux_right(ctu_view.x_flux_right_view)
    , ctu_flux_left(ctu_view.x_flux_left_view)
    {}
};

#if AETHER_DIM > 1
template<>
struct half_step_params<sweep_dir::y>{
    FaceGridY &flux_ext;
    FaceArrayView &flux_right, &flux_left, &ctu_flux_right, &ctu_flux_left;

    half_step_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view):
      flux_ext(sim.flux_y_ext)
    , flux_right(view.y_flux_right)
    , flux_left(view.y_flux_left)
    , ctu_flux_right(ctu_view.y_flux_right_view)
    , ctu_flux_left(ctu_view.y_flux_left_view)
    {}
};
#endif

#if AETHER_DIM > 2
template<>
struct half_step_params<sweep_dir::z>{
    FaceGridZ &flux_ext;
    FaceArrayView &flux_right, &flux_left, &ctu_flux_right, &ctu_flux_left;

    half_step_params(Simulation &sim, Simulation::View &view, Simulation::CTU_view &ctu_view):
      flux_ext(sim.flux_z_ext)
    , flux_right(view.z_flux_right)
    , flux_left(view.z_flux_left)
    , ctu_flux_right(ctu_view.z_flux_right_view)
    , ctu_flux_left(ctu_view.z_flux_left_view)
    {}
};
#endif


template <sweep_dir dir>
AETHER_INLINE void half_step_sweep(half_step_params<dir> &params, Simulation &sim){
    
    auto view = sim.view();
    auto &ext = view.prim.ext;
    const int nx = ext.nx;
    const int ny = ext.ny;
    const int nz = ext.nz;
    double dt = sim.time.dt;
    double dx;
    double gamma = sim.grid.gamma;

    // Loop bounds 
    int i0 = 0, i1 = nx;
    int j0 = 0, j1 = ny;
    int k0 = 0, k1 = nz;

    if constexpr (dir == sweep_dir::x) { i0 = -1; i1++; dx = sim.grid.dx;}
    if constexpr (dir == sweep_dir::y) { j0 = -1; j1++; dx = sim.grid.dy;}
    if constexpr (dir == sweep_dir::z) { k0 = -1; k1++; dx = sim.grid.dz;}    
    
    auto& FR = params.flux_right;
    auto& FL = params.flux_left;

    auto& ctu_FR = params.ctu_flux_right;
    auto& ctu_FL = params.ctu_flux_left;

    #pragma omp for collapse(3) schedule(static) 
    for (int k = k0; k < k1; ++k)
    for (int j = j0; j < j1; ++j)
    for (int i = i0; i < i1; ++i) {

        std::size_t interface_idxR = params.flux_ext.index(i-i0, j-j0, k-k0);
        std::size_t interface_idxL = params.flux_ext.index(i, j, k);
        prims L{}, R{};
        prims F_L{}, F_R{};

        L.rho = FR.comp[P::RHO][interface_idxL];
        L.vx  = FR.comp[P::VX][interface_idxL];
        L.vy  = (P::HAS_VY) ? FR.comp[P::VY][interface_idxL] : 0.0;
        L.vz  = (P::HAS_VZ) ? FR.comp[P::VZ][interface_idxL] : 0.0;
        L.p   = FR.comp[P::P][interface_idxL];

        R.rho = FL.comp[P::RHO][interface_idxR];
        R.vx  = FL.comp[P::VX][interface_idxR];
        R.vy  = (P::HAS_VY) ? FL.comp[P::VY][interface_idxR] : 0.0;
        R.vz  = (P::HAS_VZ) ? FL.comp[P::VZ][interface_idxR] : 0.0;
        R.p   = FL.comp[P::P][interface_idxR];

        half_step_update_kernel(R, L, F_R, F_L, dt, dx, gamma);

        ctu_FR.comp[P::RHO][interface_idxL] = F_L.rho;
        ctu_FR.comp[P::VX][interface_idxL] = F_L.vx;
        ctu_FR.comp[P::VY][interface_idxL] = F_L.vy;
        if constexpr (P::HAS_VZ) {
            ctu_FR.comp[P::VZ][interface_idxL] = F_L.vz;
        }
        ctu_FR.comp[P::P][interface_idxL] = F_L.p;


        ctu_FL.comp[P::RHO][interface_idxR] = F_R.rho;
        ctu_FL.comp[P::VX][interface_idxR] = F_R.vx;
        ctu_FL.comp[P::VY][interface_idxR] = F_R.vy;
        if constexpr (P::HAS_VZ) {
            ctu_FL.comp[P::VZ][interface_idxR] = F_R.vz;
        }
        ctu_FL.comp[P::P][interface_idxR] = F_R.p;
    }
};

void half_step_update(Simulation &sim){
    auto view = sim.view();
    auto ctu_view = sim.ctu_buff.view();
    #pragma omp parallel shared(sim,view,ctu_view)
    {
        half_step_params<sweep_dir::x> x_params(sim, view, ctu_view);
        half_step_sweep<sweep_dir::x>(x_params, sim);
        
        #if AETHER_DIM > 1
            half_step_params<sweep_dir::y> y_params(sim, view, ctu_view);
            half_step_sweep<sweep_dir::y>(y_params, sim);
        #endif
        #if AETHER_DIM > 2
            half_step_params<sweep_dir::z> z_params(sim, view, ctu_view);
            half_step_sweep<sweep_dir::z>(z_params, sim);
        #endif
    }
}

}