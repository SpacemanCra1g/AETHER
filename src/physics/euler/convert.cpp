#include "aether/core/simulation.hpp"
#include <aether/physics/euler/convert.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/con_layout.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>

using P = aether::prim::Prim;
using C = aether::con::Cons;

namespace loop = aether::loops;
namespace aether::physics::euler {

void cons_to_prims_domain(aether::core::Simulation &sim){
    auto domain = sim.view(); 
    auto con_array = domain.cons;
    auto prim_array = domain.prim;
    const double gamma = domain.gamma;

    Kokkos::parallel_for(
        "Cons_to_prims_domain" 
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            cons con; 
            con.rho = con_array(C::RHO,k,j,i); 
            con.mx = con_array(C::MX,k,j,i); 
            con.my = (C::HAS_MY) ? con_array(C::MY,k,j,i) : 0.0; 
            con.mz = (C::HAS_MZ) ? con_array(C::MZ,k,j,i) : 0.0; 
            con.E = con_array(C::E,k,j,i); 

            auto prim = cons_to_prims_cell(con,gamma);

            prim_array(P::RHO,k,j,i) = prim.rho;
            prim_array(P::VX,k,j,i)  = prim.vx;
            if constexpr (P::HAS_VY) prim_array(P::VY,k,j,i) = prim.vy;
            if constexpr (P::HAS_VZ) prim_array(P::VZ,k,j,i) = prim.vz;
            prim_array(P::P,k,j,i)   = prim.p;
        }
    );

}

void prims_to_cons_domain(aether::core::Simulation &sim){
    auto domain = sim.view(); 
    auto con_array = domain.cons;
    auto prim_array = domain.prim;
    const double gamma = domain.gamma;

    Kokkos::parallel_for(
        "Prims_to_cons_domain" 
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(
              [[maybe_unused]] const int k
            , [[maybe_unused]] const int j
            , const int i)
        {
            prims prim; 
            prim.rho = prim_array(P::RHO,k,j,i); 
            prim.vx = prim_array(P::VX,k,j,i); 
            prim.vy = (P::HAS_VY) ? prim_array(P::VY,k,j,i) : 0.0; 
            prim.vz = (P::HAS_VZ) ? prim_array(P::VZ,k,j,i) : 0.0; 
            prim.p = prim_array(P::P,k,j,i); 

            auto con = prims_to_cons_cell(prim,gamma);

            con_array(C::RHO,k,j,i) = con.rho;
            con_array(C::MX,k,j,i) = con.mx;
            if constexpr (C::HAS_MY) con_array(C::MY,k,j,i) = con.my;
            if constexpr (C::HAS_MZ) con_array(C::MZ,k,j,i) = con.mz;
            con_array(C::E,k,j,i) = con.E;
        }
    );
}
};
