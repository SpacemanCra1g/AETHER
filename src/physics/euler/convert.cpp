#include "aether/core/simulation.hpp"
#include <aether/physics/euler/convert.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/con_layout.hpp>
#include <cstddef>

using P = aether::prim::Prim;
using C = aether::con::Cons;

namespace aether::physics::euler {
void cons_to_prims_domain(aether::core::Simulation &sim){
    const std::size_t N = sim.ext.flat();
    auto view = sim.view(); 
    const double gamma = sim.grid.gamma;

    #pragma omp parallel for schedule(static) default(none) shared(view,N,gamma)
    for (std::size_t i = 0; i < N; ++i){
        cons con; 
        con.rho = view.cons.var(C::RHO,i); 
        con.mx = view.cons.var(C::MX,i); 
        con.my = (AETHER_DIM > 1) ? view.cons.var(C::MY,i) : 0.0; 
        con.mz = (AETHER_DIM > 2) ? view.cons.var(C::MZ,i) : 0.0; 
        con.E = view.cons.var(C::E,i); 

        auto prim = cons_to_prims_cell(con,gamma);

        view.prim.var(P::RHO,i) = prim.rho;
        view.prim.var(P::VX,i) = prim.vx;

        if constexpr (P::HAS_VY){
            view.prim.var(P::VY,i) = prim.vy;
        }
        if constexpr (P::HAS_VZ){
        view.prim.var(P::VZ,i) = prim.vz;
        }

        view.prim.var(P::P,i) = prim.p;
    }

}

void prims_to_cons_domain(aether::core::Simulation &sim){
    const std::size_t N = sim.ext.flat();
    auto view = sim.view(); 
    const double gamma = sim.grid.gamma;

    #pragma omp parallel for schedule(static) default(none) shared(view,N,gamma)
    for (std::size_t i = 0; i < N; ++i){
        prims prim; 
        prim.rho = view.prim.var(P::RHO,i); 
        prim.vx = view.prim.var(P::VX,i); 
        prim.vy = (AETHER_DIM > 1) ? view.prim.var(P::VY,i) : 0.0; 
        prim.vz = (AETHER_DIM > 2) ? view.prim.var(P::VZ,i) : 0.0; 
        prim.p = view.prim.var(P::P,i); 

        auto con = prims_to_cons_cell(prim,gamma);

        view.cons.var(C::RHO,i) = con.rho;
        view.cons.var(C::MX,i) = con.mx;
        if constexpr (C::HAS_MY){
            view.cons.var(C::MY,i) = con.my;
        }
        if constexpr (C::HAS_MZ){
        view.cons.var(C::MZ,i) = con.mz;
        }
        view.cons.var(C::E,i) = con.E;
    }
}
}