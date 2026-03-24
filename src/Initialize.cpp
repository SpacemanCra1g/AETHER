#include <Kokkos_Core.hpp>
#include "aether/core/config.hpp"
#include "aether/core/simulation.hpp"
#include <aether/core/Initialize.hpp>
#include <aether/core/enums.hpp>
#include <stdexcept>
#include <cmath>
#include <aether/core/prim_layout.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>

using P = aether::prim::Prim;
namespace loop = aether::loops;

namespace aether::core{

template<typename Sim>
static AETHER_INLINE void load_sr_shocktube(Sim &sim){
    auto domain = sim.view();
    auto prim = domain.prim;

    const double x_min = domain.x_min;
    const double x_max = domain.x_max;
    const double domain_mid = 0.5*(x_min + x_max);
    const double dx = domain.dx;
    const int ng = domain.ng;

    Kokkos::parallel_for(
        "Load SRHD shocktube initial conditions" 
      , loop::cells_interior(sim)
      , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
        const bool left = (x_min + (0.5+(i-ng))*dx < domain_mid);
        prim(P::RHO,0,0,i) = 1.0;
        prim(P::VX,0,0,i) = left ? 0.9 : 0.9;
        prim(P::VY,0,0,i) = left ? 0.9 : 0.9;
        prim(P::P,0,0,i) = left ? 1000.0 : 0.01;        
      }
    );
}

template<typename Sim>
static AETHER_INLINE void load_dmr(Sim &sim){
    auto domain = sim.view();
    auto& g = sim.grid;
    auto prim = domain.prim;
    constexpr double PI = 3.14159265358979323846;
    constexpr double inv6 = 1.0/6.0;
    const double inv_sqr3 = 1.0/std::sqrt(3.0);
    const double post_shock_u = 8.25 * std::cos(PI/6.0);
    const double post_shock_v = -8.25 * std::sin(PI/6.0);

    const double dx = domain.dx;
    const double dy = domain.dy;
    const double start_x = g.x_min + .5*dx;
    const double start_y = g.y_min + .5*dy;
    const int ng = domain.ng;

    Kokkos::parallel_for(
         "Load DMR initial conditoins"
        , loop::cells_interior(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            const double x = start_x + (i-ng)*dx;
            const double y = start_y + (j-ng)*dy;
            const bool pre_shock = (x >= inv6 + y*inv_sqr3);
            prim(P::RHO, 0, j, i) = (pre_shock) ? 1.4 : 8.0;
            prim(P::VX , 0, j, i) = (pre_shock) ? 0.0 : post_shock_u;
            prim(P::VY , 0, j, i) = (pre_shock) ? 0.0 : post_shock_v;
            prim(P::P  , 0, j, i) = (pre_shock) ? 1.0 : 116.5;
        }
    );
}

template<typename Sim, int DIM>
static AETHER_INLINE void load_sedov(Sim &sim){
    auto domain = sim.view();
    auto& g = sim.grid;
    auto prim = domain.prim;

    constexpr double PI = 3.14159265358979323846;
    const double cen_x = (g.x_min + g.x_max)*.5;
    const double cen_y = (g.y_min + g.y_max)*.5;

    const double x_start = g.x_min + g.dx*.5;
    const double y_start = g.y_min + g.dy*.5;

    const double dx = domain.dx;
    const double dy = domain.dy;
    const int ng = domain.ng;
    const double gam = domain.gamma;


    if constexpr (DIM==2){
        const double R = 3.5 *
        ((domain.dx >= domain.dy) ? domain.dy : domain.dx);
        const double R2 = R*R;
        const double blast = (gam-1.0)/(PI * R2);
        
        Kokkos::parallel_for(
             "Load 2D Sedov Inits"
            , loop::cells_interior(sim)
            , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
                const double x = x_start + (i-ng)*dx;
                const double y = y_start + (j-ng)*dy;
                const double r2 = (x-cen_x)*(x-cen_x) + (y-cen_y)*(y-cen_y);
                const bool circ = (r2 <= R2);

                prim(P::RHO ,0 ,j ,i) = 1.0;
                prim(P::VX  ,0 ,j ,i) = 0.0;
                prim(P::VY  ,0 ,j ,i) = 0.0;
                prim(P::P   ,0 ,j ,i) = (circ) ? blast : 1.e-5;
            }
        );
    } 
    else if constexpr (DIM==3) {
        const double cen_z = (domain.z_min + domain.z_max)*.5;
        const double z_start = domain.z_min + domain.dz*.5;
        const double dz = domain.dz;
        double min = (dx >= dy) ? dy : dx;
        min = (min <= dz) ? min : dz;
        const double R = 3.5 * min;
        const double R2 = R*R;
        const double R3 = R*R*R;
        const double blast = 3.0*(gam-1.0)/(4.0*(PI * R3));

        Kokkos::parallel_for(
            "Load Initial conditions for Sedov problem"
            , loop::cells_interior(sim)
            , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
            {

                const double x = x_start + (i-ng)*dx;
                const double y = y_start + (j-ng)*dy;
                const double z = z_start + (k-ng)*dz;

                const double r2 = (x-cen_x)*(x-cen_x) + (y-cen_y)*(y-cen_y) + (z-cen_z)*(z-cen_z);
                const bool circ = (r2 <= R2);

                prim(P::RHO ,k ,j ,i) = 1.0;
                prim(P::VX  ,k ,j ,i) = 0.0;
                prim(P::VY  ,k ,j ,i) = 0.0;
                prim(P::VZ  ,k ,j ,i) = 0.0;
                prim(P::P   ,k ,j ,i) = (circ) ? blast : 1.e-5;
            }
        );
    }
}

template<typename Sim>
static AETHER_INLINE void load_sod_shocktube(Sim &sim){
    
    const double domain_mid = .5*(sim.grid.x_max + sim.grid.x_min);
    const double dx = sim.grid.dx;
    const double x_min = sim.grid.x_min;

    auto domain = sim.view();
    auto prim = domain.prim;
    const int ng = domain.ng;

    Kokkos::parallel_for(
        "load_1D_sod_initial_conditions"
        , loop::cells_interior(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            const bool left = x_min + (0.5 + (i-ng))*dx < domain_mid;
            prim(P::RHO,0,0,i) = left ? 1.0 : 0.125;
            prim(P::VX,0,0,i) = 0.0;
            prim(P::P,0,0,i) = left ? 1.0 : 0.1;
            if constexpr (P::HAS_VY) prim(P::VY,0,0,i) = 0.0;
            if constexpr (P::HAS_VZ) prim(P::VZ,0,0,i) = 0.0;
        }
    );
}

template<typename Sim>
static AETHER_INLINE void load_sod_y(Sim &sim){
    auto domain = sim.view();
    auto& g = sim.grid;
    const double domain_mid = .5*(g.y_max + g.y_min);
    const double dy = domain.dy;
    const double y_min = g.y_min;
    const int ng = domain.ng;
    auto prim = domain.prim;

    Kokkos::parallel_for(
        "Load Sod_y initial conditions"
        , loop::cells_interior(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            const bool left = (y_min + (0.5 + (j-ng))*dy < domain_mid);
            prim(P::RHO, 0, j, i) = (left) ? 1.0 : 0.125;
            prim(P::VX , 0, j, i) = 0.0;
            prim(P::P  , 0, j, i) = (left) ? 1.0 : 0.1;
        }
    );
}

template<typename Sim>
static AETHER_INLINE void load_sod_z(Sim &sim){
    auto domain = sim.view();
    auto prim = domain.prim;
    const double domain_mid = .5*(domain.z_max + domain.z_min);
    const double dz = domain.dz;
    const double z_min = domain.z_min;
    const int ng = domain.ng;
    
    Kokkos::parallel_for(
        "Load Sod_z initial conditions"
        , loop::cells_interior(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            const bool left = (z_min + (0.5+ (k-ng))*dz < domain_mid);
            prim(P::RHO, k, j, i) = (left) ? 1.0 : 0.125;
            prim(P::VX , k, j, i) = 0.0;
            prim(P::P  , k, j, i) = (left) ? 1.0 : 0.1;
        }
    );
}

template <typename Sim>
void initialize_domain(Sim &sim){
    if constexpr (AETHER_PHYSICS_EULER){
        if constexpr (Sim::dim == 1){ 
            switch (sim.cfg.prob) {                
                case test_problem::sod : load_sod_shocktube(sim); break;
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
        else if constexpr (Sim::dim == 2){
            switch (sim.cfg.prob) {                
                case test_problem::sedov : load_sedov<Sim, Sim::dim>(sim); break;
                case test_problem::dmr : load_dmr(sim); break;                
                case test_problem::sod_y : load_sod_y(sim); break;                
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
        else if constexpr (Sim::dim == 3){
            switch (sim.cfg.prob) {                
                case test_problem::sod_z : load_sod_z(sim); break;                
                case test_problem::sedov : load_sedov<Sim, AETHER_DIM>(sim); break;
                default: throw std::runtime_error("Invalid Test problem config");
            };
        }
    }
    else if constexpr (AETHER_PHYSICS_SRHD){
        if constexpr (Sim::dim == 1){ 
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
