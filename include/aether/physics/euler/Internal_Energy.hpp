#pragma once
#include "aether/core/con_layout.hpp"
#include "aether/core/prim_layout.hpp"
#include <aether/core/config.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>

#define USE_ENERGY_CORRECT true

namespace loop = aether::loops;
namespace aether::core{
    
using P = aether::prim::Prim;
using C = aether::con::Cons;


template<class Sim, class CellView>
AETHER_INLINE 
void Update_internal_energy(CellView cons, CellView prims, Sim& sim){

    #if USE_ENERGY_CORRECT
        Kokkos::parallel_for(
          "internal energy update"
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(const int k, const int j, const int i) {
            double rho = cons(C::RHO,k,j,i);
            double vx  = cons(C::MX,k,j,i) / rho;
            double vy  = (C::HAS_MY) ? cons(C::MY,k,j,i) / rho : 0.0;
            double vz  = (C::HAS_MZ) ? cons(C::MZ,k,j,i) / rho : 0.0;
            double v2 = vx*vx + vy*vy + vz*vz;
            double kin_e_rho = rho*.5*v2;

            // This is a temp total domain replacement of total energy
            // cons(C::E,k,j,i) = prims(P::EINT,k,j,i) + kin_e_rho;
            // prims(P::EINT,k,j,i) = (cons(C::E,k,j,i) - kin_e_rho)/rho;
        }
        );
    #endif

}

template<class Sim, class CellView>
AETHER_INLINE 
void Initialize_internal_energy(CellView prims, Sim& sim){

        Kokkos::parallel_for(
          "internal energy update"
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(const int k, const int j, const int i) {
            prims(P::EINT,k,j,i) = prims(P::P,k,j,i) / (sim.grid.gamma - 1.0);
        }
        );

}

}