#pragma once
#include "aether/core/prim_layout.hpp"
#include "aether/physics/euler/convert.hpp"
#include <aether/core/config.hpp>
#include <aether/physics/api.hpp>
#include <iostream>
#include <cmath>

using prims = aether::phys::prims;
using cons = aether::phys::cons;
using P = aether::prim::Prim;

namespace aether::core{

AETHER_INLINE void half_step_update_kernel(
      prims &in_R, prims &in_L
    , prims &out_R, prims &out_L
    , const double dt, const double dx
    , const double gamma){

    cons in_Rc = prims_to_cons_cell(in_R, gamma);
    cons in_Lc = prims_to_cons_cell(in_L, gamma);

    const double dtx = .5*dt/dx;

    prims F_R = flux_from_prim_cell(in_R, gamma);
    prims F_L = flux_from_prim_cell(in_L, gamma);


    cons out_Lc, out_Rc;

    out_Lc.rho = in_Lc.rho - dtx*(F_R.rho - F_L.rho);
    out_Lc.mx = in_Lc.mx - dtx*(F_R.vx - F_L.vx);
    out_Lc.my = in_Lc.my - dtx*(F_R.vy - F_L.vy);    
    out_Lc.mz = in_Lc.mz - dtx*(F_R.vz - F_L.vz);    
    out_Lc.E = in_Lc.E - dtx*(F_R.p - F_L.p);    

    out_Rc.rho = in_Rc.rho - dtx*(F_R.rho - F_L.rho);
    out_Rc.mx = in_Rc.mx - dtx*(F_R.vx - F_L.vx);
    out_Rc.my = in_Rc.my - dtx*(F_R.vy - F_L.vy);    
    out_Rc.mz = in_Rc.mz - dtx*(F_R.vz - F_L.vz);    
    out_Rc.E = in_Rc.E - dtx*(F_R.p - F_L.p);    

    if (!std::isfinite(out_Lc.E) || out_Lc.E <= 0.0){
      std::cout << "negative Energy right: " << out_Lc.E;
      exit(0);
    }

    out_R = cons_to_prims_cell(out_Rc, gamma);
    out_L = cons_to_prims_cell(out_Lc, gamma);

    if (!std::isfinite(out_R.p) || out_R.p <= 0.0){
      std::cout << "negative pressure right: " << out_R.p;
      exit(0);
    }

    if (!std::isfinite(out_L.p) || out_L.p <= 0.0){
      std::cout << "negative pressure left: " << out_L.p;
      exit(0);
    }
}

void half_step_update(Simulation &sim);

}

