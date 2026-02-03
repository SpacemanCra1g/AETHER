#include "aether/core/config.hpp"
#include "aether/core/simulation.hpp"
#include <aether/core/Initialize.hpp>
#include <aether/core/enums.hpp>
#include <stdexcept>
#include <cmath>
#include <aether/core/prim_layout.hpp>

using P = aether::prim::Prim;

template<typename Sim>
static AETHER_INLINE void load_sr_shocktube(Sim &sim){
    typename Sim::View domain = sim.view();
    bool left;
    const double domain_mid = .5*(sim.grid.x_max + sim.grid.x_min);
    const double dx = sim.grid.dx;
    const double x_min = sim.grid.x_min;
    const int nx = sim.grid.nx;
    
    #pragma omp parallel for private(left) schedule(static)
    for (int i = 0; i < nx; ++i){
        left = (x_min + (0.5+ i)*dx < domain_mid);

        domain.prim.var(P::RHO, i, 0, 0) = 1.0;
        domain.prim.var(P::VX , i, 0, 0) = 0.9;
        domain.prim.var(P::VY , i, 0, 0) = 0.9;
        domain.prim.var(P::VZ , i, 0, 0) = 0.0;
        domain.prim.var(P::P  , i, 0, 0) = (left) ? 1000.0 : 0.01;
    }
}

template<typename Sim>
static AETHER_INLINE void load_dmr(Sim &sim){
    auto view = sim.view();
    auto &grid = sim.grid;
    constexpr double PI = 3.14159265358979323846;
    const double inv6 = 1.0/6.0;
    const double inv_sqr3 = 1.0/std::sqrt(3.0);
    const double post_shock_u = 8.25 * std::cos(PI/6.0);
    const double post_shock_v = -8.25 * std::sin(PI/6.0);

    const int nx = view.nx;
    const int ny = view.ny;
    const double dx = view.dx;
    const double dy = view.dy;
    const double start_x = grid.x_min + .5*dx;
    const double start_y = grid.y_min + .5*dy;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < nx; ++i){
        for (int j = 0; j < ny; ++j){
            const double x = start_x + i*dx;
            const double y = start_y + j*dy;
            const bool pre_shock = (x >= inv6 + y*inv_sqr3);
            view.prim.var(P::RHO, i, j, 0) = (pre_shock) ? 1.4 : 8.0;
            view.prim.var(P::VX , i, j, 0) = (pre_shock) ? 0.0 : post_shock_u;
            view.prim.var(P::VY , i, j, 0) = (pre_shock) ? 0.0 : post_shock_v;
            view.prim.var(P::P  , i, j, 0) = (pre_shock) ? 1.0 : 116.5;
        }
    }
}

template<typename Sim, int DIM>
static AETHER_INLINE void load_sedov(Sim &sim){
    typename Sim::View domain = sim.view();
    auto &grid = sim.grid;

    constexpr double PI = 3.14159265358979323846;
    const double cen_x = (grid.x_min + grid.x_max)*.5;
    const double cen_y = (grid.y_min + grid.y_max)*.5;

    const double x_start = grid.x_min + grid.dx*.5;
    const double y_start = grid.y_min + grid.dy*.5;

    const double dx = grid.dx;
    const double dy = grid.dy;
    const int nx = grid.nx;
    const int ny = grid.ny;
    const double gam = grid.gamma;


    if constexpr (DIM==2){
        const double R = 3.5 *
        ((grid.dx >= grid.dy) ? grid.dy : grid.dx);
        const double R2 = R*R;
        const double blast = (gam-1.0)/(PI * R2);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < ny; ++j){
                const double x = x_start + i*dx;
                const double y = y_start + j*dy;
                const double r2 = (x-cen_x)*(x-cen_x) + (y-cen_y)*(y-cen_y);
                const bool circ = (r2 <= R2);

                domain.prim.var(P::RHO ,i ,j ,0) = 1.0;
                domain.prim.var(P::VX  ,i ,j ,0) = 0.0;
                domain.prim.var(P::VY  ,i ,j ,0) = 0.0;
                domain.prim.var(P::P   ,i ,j ,0) = (circ) ? blast : 1.e-5;
            }
        }
    } 
    else if constexpr (DIM==3) {
        const double cen_z = (grid.z_min + grid.z_max)*.5;
        const double z_start = grid.z_min + grid.dz*.5;
        const double dz = grid.dz;
        const int nz = grid.nz;
        double min = (grid.dx >= grid.dy) ? grid.dy : grid.dx;
        min = (min <= grid.dz) ? min : grid.dz;
        const double R = 3.5 * min;
        const double R2 = R*R;
        const double R3 = R*R*R;
        const double blast = 3.0*(gam-1.0)/(4.0*(PI * R3));

        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < ny; ++j){
                for (int k = 0; k < nz; ++k){

                    const double x = x_start + i*dx;
                    const double y = y_start + j*dy;
                    const double z = z_start + k*dz;

                    const double r2 = (x-cen_x)*(x-cen_x) + (y-cen_y)*(y-cen_y) + (z-cen_z)*(z-cen_z);
                    const bool circ = (r2 <= R2);

                    domain.prim.var(P::RHO ,i ,j ,k) = 1.0;
                    domain.prim.var(P::VX  ,i ,j ,k) = 0.0;
                    domain.prim.var(P::VY  ,i ,j ,k) = 0.0;
                    domain.prim.var(P::VZ  ,i ,j ,k) = 0.0;
                    domain.prim.var(P::P   ,i ,j ,k) = (circ) ? blast : 1.e-5;
                }
            }
        }
    }
}


template<typename Sim>
static AETHER_INLINE void load_sod_shocktube(Sim &sim){
    typename Sim::View domain = sim.view();
    bool left;
    const double domain_mid = .5*(sim.grid.x_max + sim.grid.x_min);
    const double dx = sim.grid.dx;
    const double x_min = sim.grid.x_min;
    const int nx = sim.grid.nx;
    
    #pragma omp parallel for private(left) schedule(static)
    for (int i = 0; i < nx; ++i){
        left = (x_min + (0.5+ i)*dx < domain_mid);

        domain.prim.var(P::RHO, i, 0, 0) = (left) ? 1.0 : 0.125;
        domain.prim.var(P::VX , i, 0, 0) = 0.0;
        domain.prim.var(P::P  , i, 0, 0) = (left) ? 1.0 : 0.1;
    }
}

template<typename Sim>
static AETHER_INLINE void load_sod_y(Sim &sim){
    typename Sim::View domain = sim.view();
    bool left;
    const double domain_mid = .5*(sim.grid.y_max + sim.grid.y_min);
    const double dy = sim.grid.dy;
    const double y_min = sim.grid.y_min;
    const int ny = sim.grid.ny;
    const int nx = sim.grid.nx;
    
    #pragma omp parallel for private(left) schedule(static)
    for (int j = 0; j < ny; ++j){
        left = (y_min + (0.5+ j)*dy <= domain_mid);
        #pragma omp simd
        for (int i = 0; i < nx; ++i){
            domain.prim.var(P::RHO, i, j, 0) = (left) ? 1.0 : 0.125;
            domain.prim.var(P::VX , i, j, 0) = 0.0;
            domain.prim.var(P::P  , i, j, 0) = (left) ? 1.0 : 0.1;
        }
    }
}

namespace aether::core {
using Sim = Simulation;

template <typename Sim>
void initialize_domain(Sim &sim){
    if constexpr (AETHER_PHYSICS_EULER){
        if constexpr (AETHER_DIM==1){ 
            switch (sim.cfg.prob) {                
                case test_problem::sod : load_sod_shocktube(sim); break;
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
        else if constexpr (AETHER_DIM==2){
            switch (sim.cfg.prob) {                
                case test_problem::sedov : load_sedov<Sim, AETHER_DIM>(sim); break;
                case test_problem::dmr : load_dmr(sim); break;                
                case test_problem::sod_y : load_sod_y(sim); break;                
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
    }
    else if constexpr (AETHER_PHYSICS_SRHD){
        if constexpr (AETHER_DIM==1){ 
            switch (sim.cfg.prob) {                
                case test_problem::sr_shocktube : load_sr_shocktube(sim); break;
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
    }
    else if constexpr (AETHER_PHYSICS_MHD){
        // Will be populated later with other test problems
    }
}

template void initialize_domain<Simulation>(Simulation &);
}