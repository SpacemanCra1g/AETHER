#pragma once
#include "aether/core/simulation.hpp"
#include <aether/core/config.hpp>
#include <aether/physics/euler/variable_structs.hpp>

namespace aether::physics::euler{
    AETHER_INLINE cons prims_to_cons_cell(prims& p, const double& gamma){
        cons U;
        const double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        U.rho = p.rho;
        U.mx = p.vx*p.rho;
        U.my = p.vy*p.rho;
        U.mz = p.vz*p.rho;
        U.E = p.p/(gamma-1.0) + 0.5*p.rho*v2;
        return U;
    }

    AETHER_INLINE prims cons_to_prims_cell(cons& c,const double& gamma){
        prims V;
        V.rho = c.rho;
        const double inv_rho = 1.0/c.rho;
        V.vx = c.mx*inv_rho;
        V.vy = c.my*inv_rho;
        V.vz = c.mz*inv_rho;
        const double v2 = V.vx*V.vx + V.vy*V.vy + V.vz*V.vz;
        V.p = (gamma-1.0)*(c.E - 0.5*c.rho*v2);

        return V;
    }

    void cons_to_prims_domain(aether::core::Simulation &sim);

    void prims_to_cons_domain(aether::core::Simulation &sim);
}
