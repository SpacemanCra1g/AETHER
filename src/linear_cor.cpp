#include "aether/core/Kokkos_loopBounds.hpp"
#include "aether/core/con_layout.hpp"
#include "aether/physics/euler/variable_structs.hpp"
#include <aether/physics/api.hpp>


namespace aether::core{
using C = aether::con::Cons;

void correct_domain(Simulation& sim){
    auto view = sim.view();
    auto prims = view.prim;
    auto cons = view.cons;
    const double gamma = sim.grid.gamma;

    Kokkos::parallel_for(
        "Conservative correction",
        aether::loops::cells_interior(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i){
            aether::phys::prims p, pl, pr;
            aether::phys::cons c;

            p.rho = prims(P::RHO, k , j, i);
            p.vx  = prims(P::VX , k , j, i);            
            p.p   = prims(P::P , k , j, i);            
            p.vy  = (P::HAS_VY) ? prims(P::VY , k , j, i) : 0.0;            
            p.vz  = (P::HAS_VZ) ? prims(P::VZ , k , j, i) : 0.0;            

            pl.rho = prims(P::RHO, k , j, i-1);
            pl.vx  = prims(P::VX , k , j, i-1);            
            pl.p   = prims(P::P , k , j, i-1);            
            pl.vy  = (P::HAS_VY) ? prims(P::VY , k , j, i-1) : 0.0;            
            pl.vz  = (P::HAS_VZ) ? prims(P::VZ , k , j, i-1) : 0.0;            

            pr.rho = prims(P::RHO, k , j, i+1);
            pr.vx  = prims(P::VX , k , j, i+1);            
            pr.p   = prims(P::P , k , j, i+1);            
            pr.vy  = (P::HAS_VY) ? prims(P::VY , k , j, i+1) : 0.0;            
            pr.vz  = (P::HAS_VZ) ? prims(P::VZ , k , j, i+1) : 0.0;            

            c.rho = cons(C::RHO, k , j, i);
            c.mx  = cons(C::MX , k , j, i);            
            c.E   = cons(C::E  , k , j, i);            
            c.my  = (C::HAS_MY) ? cons(C::MY , k , j, i) : 0.0;            
            c.mz  = (C::HAS_MZ) ? cons(C::MZ , k , j, i) : 0.0;    
            
            auto C_cor = linear_correction(c,pl,p,pr,gamma,view,k,j,i);

            cons(C::RHO, k , j, i) = C_cor.rho;
            cons(C::MX , k , j, i) = C_cor.mx;
            cons(C::E  , k , j, i) = C_cor.E;
            if constexpr (C::HAS_MY) cons(C::MY , k , j, i) = C_cor.my;
            if constexpr (C::HAS_MZ) cons(C::MZ , k , j, i) = C_cor.mz;
        }
    );
}

};