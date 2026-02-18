#pragma once
#include "aether/core/views.hpp"
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/enums.hpp>
#include <stdexcept>

using P = aether::prim::Prim;
namespace aether::core {


template<sweep_dir dir, class V>
struct RiemannIO;

// These are the x-sweep IO templates
template<>
struct RiemannIO<sweep_dir::x, Simulation::View>{
    static auto flux(Simulation::View& v) -> FaceArrayView& {return v.x_flux;}
    static auto flux_left(Simulation::View& v) -> FaceArrayView& {return v.x_flux_left;}
    static auto flux_right(Simulation::View& v) -> FaceArrayView& {return v.x_flux_right;}
    static auto flux_ext(Simulation& sim) -> FaceGridX& {return sim.flux_x_ext;}
};

template<>
struct RiemannIO<sweep_dir::x, Simulation::CTU_view>{
    static auto flux(Simulation::CTU_view& v) -> FaceArrayView& {return v.x_flux_view;}
    static auto flux_left(Simulation::CTU_view& v) -> FaceArrayView& {return v.x_flux_left_view;}
    static auto flux_right(Simulation::CTU_view& v) -> FaceArrayView& {return v.x_flux_right_view;}
    static auto flux_ext(Simulation& sim) -> FaceGridX& {return sim.flux_x_ext;}
};

// y-sweep IO templates
#if AETHER_DIM > 1
template<>
struct RiemannIO<sweep_dir::y, Simulation::View>{
    static auto flux(Simulation::View& v) -> FaceArrayView& {return v.y_flux;}
    static auto flux_left(Simulation::View& v) -> FaceArrayView& {return v.y_flux_left;}
    static auto flux_right(Simulation::View& v) -> FaceArrayView& {return v.y_flux_right;}
    static auto flux_ext(Simulation& sim) -> FaceGridY& {return sim.flux_y_ext;}
};

template<>
struct RiemannIO<sweep_dir::y, Simulation::CTU_view>{
    static auto flux(Simulation::CTU_view& v) -> FaceArrayView& {return v.y_flux_view;}
    static auto flux_left(Simulation::CTU_view& v) -> FaceArrayView& {return v.y_flux_left_view;}
    static auto flux_right(Simulation::CTU_view& v) -> FaceArrayView& {return v.y_flux_right_view;}
    static auto flux_ext(Simulation& sim) -> FaceGridY& {return sim.flux_y_ext;}
};
#endif

// z-sweep IO templates
#if AETHER_DIM > 2
template<>
struct RiemannIO<sweep_dir::z, Simulation::View>{
    static auto flux(Simulation::View& v) -> FaceArrayView& {return v.z_flux;}
    static auto flux_left(Simulation::View& v) -> FaceArrayView& {return v.z_flux_left;}
    static auto flux_right(Simulation::View& v) -> FaceArrayView& {return v.z_flux_right;}
    static auto flux_ext(Simulation& sim) -> FaceGridZ& {return sim.flux_z_ext;}
};

template<>
struct RiemannIO<sweep_dir::z, Simulation::CTU_view>{
    static auto flux(Simulation::CTU_view& v) -> FaceArrayView& {return v.z_flux_view;}
    static auto flux_left(Simulation::CTU_view& v) -> FaceArrayView& {return v.z_flux_left_view;}
    static auto flux_right(Simulation::CTU_view& v) -> FaceArrayView& {return v.z_flux_right_view;}
    static auto flux_ext(Simulation& sim) -> FaceGridZ& {return sim.flux_z_ext;}
};
#endif

template<sweep_dir dir>
struct VelMap;

template<> struct VelMap<sweep_dir::x> {
    static constexpr int VN  = P::VX;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VZ;
};

template<> struct VelMap<sweep_dir::y> {
    static constexpr int VN  = P::VY;
    static constexpr int VT1 = P::VX;
    static constexpr int VT2 = P::VZ;
};

template<> struct VelMap<sweep_dir::z> {
    static constexpr int VN  = P::VZ;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VX;
};

template<sweep_dir dir, class V>
struct riemann_sweep_params;

template<class V> struct riemann_sweep_params<sweep_dir::x,V>{
FaceArrayView &flux_right, &flux_left, &flux;
FaceGridX &flux_ext;
riemann_sweep_params(Simulation &sim, V& v):
      flux_right(RiemannIO<sweep_dir::x, V>::flux_right(v))
    , flux_left(RiemannIO<sweep_dir::x, V>::flux_left(v))
    , flux(RiemannIO<sweep_dir::x, V>::flux(v))
    , flux_ext(RiemannIO<sweep_dir::x, V>::flux_ext(sim))
    {}
};

template<class V> struct riemann_sweep_params<sweep_dir::y,V>{
FaceArrayView &flux_right, &flux_left, &flux;
FaceGridY &flux_ext;
riemann_sweep_params(Simulation &sim, V& v):
      flux_right(RiemannIO<sweep_dir::y, V>::flux_right(v))
    , flux_left(RiemannIO<sweep_dir::y, V>::flux_left(v))
    , flux(RiemannIO<sweep_dir::y, V>::flux(v))
    , flux_ext(RiemannIO<sweep_dir::y, V>::flux_ext(sim))
    {}
};

template<class V> struct riemann_sweep_params<sweep_dir::z,V>{
FaceArrayView &flux_right, &flux_left, &flux;
FaceGridZ &flux_ext;
riemann_sweep_params(Simulation &sim, V& v):
      flux_right(RiemannIO<sweep_dir::z, V>::flux_right(v))
    , flux_left(RiemannIO<sweep_dir::z, V>::flux_left(v))
    , flux(RiemannIO<sweep_dir::z, V>::flux(v))
    , flux_ext(RiemannIO<sweep_dir::z, V>::flux_ext(sim))
    {}
};



template<riemann solv, sweep_dir dir, class V>
AETHER_INLINE void Riemann_sweep(Simulation& Sim, riemann_sweep_params<dir,V> params) noexcept {
    
    const double gamma = Sim.grid.gamma;
    auto view = Sim.view();
    auto ext  = view.prim.ext;
    const int nx = ext.nx, ny = ext.ny, nz = ext.nz;

    // Loop bounds
    int i0 = 0, i1 = nx;
    int j0 = 0, j1 = ny;
    int k0 = 0, k1 = nz;

    if constexpr (dir == sweep_dir::x) { i0 = -1; }
    if constexpr (dir == sweep_dir::y) { j0 = -1; }
    if constexpr (dir == sweep_dir::z) { k0 = -1; }

    auto& FR = params.flux_right;
    auto& FL = params.flux_left;
    auto& Flux = params.flux;

    #pragma omp for collapse(3) schedule(static) nowait
    for (int k = k0; k < k1; ++k)
    for (int j = j0; j < j1; ++j)
    for (int i = i0; i < i1; ++i) {

        std::size_t interface_idx = params.flux_ext.index(i-i0, j-j0, k-k0);

        aether::phys::prims L{}, R{}, F{};

        L.rho = FL.comp[P::RHO][interface_idx];
        L.vx  = FL.comp[VelMap<dir>::VN][interface_idx];
        L.vy  = (P::HAS_VY) ? FL.comp[VelMap<dir>::VT1][interface_idx] : 0.0;
        L.vz  = (P::HAS_VZ) ? FL.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
        L.p   = FL.comp[P::P][interface_idx];

        R.rho = FR.comp[P::RHO][interface_idx];
        R.vx  = FR.comp[VelMap<dir>::VN][interface_idx];
        R.vy  = (P::HAS_VY) ? FR.comp[VelMap<dir>::VT1][interface_idx] : 0.0;
        R.vz  = (P::HAS_VZ) ? FR.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
        R.p   = FR.comp[P::P][interface_idx];


        if constexpr (solv == riemann::hll) {
            F = hll(L, R, gamma);
        }

        Flux.comp[P::RHO][interface_idx] = F.rho;
        Flux.comp[VelMap<dir>::VN][interface_idx]  = F.vx;
        if constexpr (P::HAS_VY) Flux.comp[VelMap<dir>::VT1][interface_idx] = F.vy;
        if constexpr (P::HAS_VZ) Flux.comp[VelMap<dir>::VT2][interface_idx] = F.vz;
        Flux.comp[P::P][interface_idx]   = F.p;
    }
}



template<class T>
void Riemann_dispatch(Simulation& Sim, T &V){
    riemann_sweep_params<sweep_dir::x, T> params_x(Sim,V);
    #if AETHER_DIM > 1
    riemann_sweep_params<sweep_dir::y, T> params_y(Sim,V);
    #endif
    #if AETHER_DIM > 2
    riemann_sweep_params<sweep_dir::z, T> params_z(Sim,V);
    #endif
    
    switch (Sim.cfg.riem) {
        case riemann::hll:{
            #if AETHER_DIM == 1
            #pragma omp parallel default(none) shared(Sim,V,params_x)
            {   
                Riemann_sweep<riemann::hll,sweep_dir::x>(Sim,params_x);
            }
            #elif AETHER_DIM == 2 
            #pragma omp parallel default(none) shared(Sim,V,params_x,params_y)
            {
                Riemann_sweep<riemann::hll,sweep_dir::x>(Sim,params_x);
                Riemann_sweep<riemann::hll,sweep_dir::y>(Sim,params_y);
            }
            #elif AETHER_DIM == 3
            #pragma omp parallel default(none) shared(Sim,V,params_x,params_y,params_z)
            {
                Riemann_sweep<riemann::hll,sweep_dir::x>(Sim,params_x);
                Riemann_sweep<riemann::hll,sweep_dir::y>(Sim,params_y);
                Riemann_sweep<riemann::hll,sweep_dir::z>(Sim,params_z);
            }
            #endif
            break;
        }
        default:
            throw std::runtime_error("RiemannSolve: unknown space solver, Dispatch");
            break;
    }
}


}