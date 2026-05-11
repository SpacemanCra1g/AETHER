#pragma once
#include "aether/core/con_layout.hpp"
#include "aether/core/prim_layout.hpp"
#include <aether/core/config.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>

#define FORCE_ENERGY_CORRECT false
#define NEVER_ENERGY_CORRECT false


namespace loop = aether::loops;
namespace aether::core{
    
using P = aether::prim::Prim;
using C = aether::con::Cons;


template<class Sim, class CellView>
AETHER_INLINE 
void Update_internal_energy(CellView cons, CellView prims, Sim& sim){
    double eintSwitch = sim.cfg.eintSwitch;

    Kokkos::parallel_for(
        "internal energy update"
    , loop::cells_full(sim)
    , KOKKOS_LAMBDA(const int k, const int j, const int i) {
        double rho = cons(C::RHO,k,j,i);
        double vx  = cons(C::MX,k,j,i) / rho;
        double vy  = (C::HAS_MY) ? cons(C::MY,k,j,i) / rho : 0.0;
        double vz  = (C::HAS_MZ) ? cons(C::MZ,k,j,i) / rho : 0.0;
        double v2 = vx*vx + vy*vy + vz*vz;
        double kin_e = 0.5*v2;
        double eint = prims(P::EINT,k,j,i) / rho;

        // This is a temp total domain replacement of total energy
        if ( (eint < eintSwitch * kin_e || FORCE_ENERGY_CORRECT) && !NEVER_ENERGY_CORRECT ){
            cons(C::E,k,j,i) = prims(P::EINT,k,j,i) + rho*kin_e;
            
        }
        else{
            prims(P::EINT,k,j,i) = cons(C::E,k,j,i) - kin_e*rho;
        }
    }
    );
}

template<class Sim, class CellView>
AETHER_INLINE 
void Initialize_internal_energy(CellView prims, Sim& sim){

        Kokkos::parallel_for(
          "Initilize internal energy"
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(const int k, const int j, const int i) {
            prims(P::EINT,k,j,i) = prims(P::P,k,j,i) / ((sim.grid.gamma - 1.0));
        }
        );

}

template<class Sim, class CellView>
AETHER_INLINE 
void spec_to_internal_energy(CellView prims, Sim& sim){
        Kokkos::parallel_for(
          "Convert specific internal energy to internal en"
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(const int k, const int j, const int i) {
            prims(P::EINT,k,j,i) /= prims(P::RHO,k,j,i);
        }
        );
}

template<class Sim, class CellView>
AETHER_INLINE 
void internal_energy_to_specific(CellView prims, Sim& sim){
        Kokkos::parallel_for(
          "Convert internal energy to specific internal energy"
        , loop::cells_full(sim)
        , KOKKOS_LAMBDA(const int k, const int j, const int i) {
            prims(P::EINT,k,j,i) *= prims(P::RHO,k,j,i);
        }
        );
}

}