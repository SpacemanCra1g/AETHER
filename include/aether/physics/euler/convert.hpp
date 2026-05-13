#pragma once
#include "Kokkos_Macros.hpp"
#include "aether/core/simulation.hpp"
#include <aether/core/config.hpp>
#include <aether/physics/euler/variable_structs.hpp>

namespace aether::physics::euler{
    KOKKOS_INLINE_FUNCTION
    cons prims_to_cons_cell(prims& p, double gamma){
        cons U;
        const double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        U.rho = p.rho;
        U.mx = p.vx*p.rho;
        U.my = p.vy*p.rho;
        U.mz = p.vz*p.rho;
        U.E = p.p/(gamma-1.0) + 0.5*p.rho*v2;
        U.rho_e = p.e ;
        return U;
    }

    KOKKOS_INLINE_FUNCTION
    cons prims_to_cons_cell(const prims& p, double gamma){
        cons U;
        const double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        U.rho = p.rho;
        U.mx = p.vx*p.rho;
        U.my = p.vy*p.rho;
        U.mz = p.vz*p.rho;
        U.E = p.p/(gamma-1.0) + 0.5*p.rho*v2;
        U.rho_e = p.e;
        return U;
    }
    KOKKOS_INLINE_FUNCTION
    prims cons_to_prims_cell(cons& c, double gamma){
        prims V;
        V.rho = c.rho;
        const double inv_rho = 1.0/c.rho;
        V.vx = c.mx*inv_rho;
        V.vy = c.my*inv_rho;
        V.vz = c.mz*inv_rho;
        const double v2 = V.vx*V.vx + V.vy*V.vy + V.vz*V.vz;
        V.p = (gamma-1.0)*(c.E - 0.5*c.rho*v2);
        V.e = c.rho_e ;
        return V;
    }

    KOKKOS_INLINE_FUNCTION
    prims cons_to_prims_cell(const cons& c, double gamma){
        prims V;
        V.rho = c.rho;
        const double inv_rho = 1.0/c.rho;
        V.vx = c.mx*inv_rho;
        V.vy = c.my*inv_rho;
        V.vz = c.mz*inv_rho;
        const double v2 = V.vx*V.vx + V.vy*V.vy + V.vz*V.vz;
        V.p = (gamma-1.0)*(c.E - 0.5*c.rho*v2);
        V.e = c.rho_e;
        return V;
    }

    KOKKOS_INLINE_FUNCTION
    prims flux_from_prim_cell(prims &W, double gamma){
        prims F;
        double inv_gm1 = 1.0/(gamma-1.0);
        const double v2 = W.vx*W.vx + W.vy*W.vy + W.vz*W.vz;
        const double E  = W.p * inv_gm1 + 0.5 * W.rho * v2;

        F.rho = W.rho * W.vx;
        F.vx  = W.rho * W.vx * W.vx + W.p;
        F.vy  = W.rho * W.vx * W.vy;
        F.vz  = W.rho * W.vx * W.vz;
        F.p   = W.vx * (E + W.p);
        F.e   = F.rho / W.rho;
        return F;
    }

    KOKKOS_INLINE_FUNCTION
    prims flux_from_prim_cell(const prims &W, double gamma){
        prims F;
        double inv_gm1 = 1.0/(gamma-1.0);
        const double v2 = W.vx*W.vx + W.vy*W.vy + W.vz*W.vz;
        const double E  = W.p * inv_gm1 + 0.5 * W.rho * v2;

        F.rho = W.rho * W.vx;
        F.vx  = W.rho * W.vx * W.vx + W.p;
        F.vy  = W.rho * W.vx * W.vy;
        F.vz  = W.rho * W.vx * W.vz;
        F.p   = W.vx * (E + W.p);
        F.e   = F.rho / W.rho;
        return F;
    }

    void cons_to_prims_domain(aether::core::Simulation &sim);

    void prims_to_cons_domain(aether::core::Simulation &sim);
}
