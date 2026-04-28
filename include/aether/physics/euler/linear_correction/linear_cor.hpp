#pragma once
#include "aether/core/simulation.hpp"
#include "aether/physics/euler/convert.hpp"
#include <aether/core/config.hpp>
#include <aether/physics/api.hpp>
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <cmath>

namespace aether::physics::euler {

static constexpr bool APPLY_LINEAR_CORRECTION_EVERYWHERE = true;
static constexpr double CONTACT_TOL_P  = 1.0e-8;
static constexpr double CONTACT_TOL_VX = 1.0e-8;
static constexpr double DELTA_E_TRIGGER = 1.0e-5;

KOKKOS_INLINE_FUNCTION
bool small_stencil_change(const prims& PL,
                          const prims& Pc,
                          const prims& PR) noexcept {
    const double p_scale  = fmax(fmax(fabs(PL.p),  fabs(Pc.p)),  fabs(PR.p));
    const double vx_scale = fmax(fmax(fabs(PL.vx), fabs(Pc.vx)), fabs(PR.vx));

    const double dp  = fmax(fabs(Pc.p  - PL.p),  fabs(PR.p  - Pc.p)) / fmax(p_scale,  1.0e-14);
    const double dvx = fmax(fabs(Pc.vx - PL.vx), fabs(PR.vx - Pc.vx)) / fmax(vx_scale, 1.0e-14);

    return (dp < CONTACT_TOL_P) && (dvx < CONTACT_TOL_VX);
}

KOKKOS_INLINE_FUNCTION
double deltaE_pair(const prims& PL,
                   const prims& PR,
                   const double dt_dx,
                   const double u0) noexcept {
    const double theta = u0 * dt_dx;

    const double a = (1.0 - theta) * PR.rho;
    const double b = theta * PL.rho;
    const double denom = a + b;

    const double term1 = a * PR.vy * PR.vy + b * PL.vy * PL.vy;
    const double term2 = (a * PR.vy + b * PL.vy);

    return 0.5 * (term1 - (term2 * term2) / denom);
}

KOKKOS_INLINE_FUNCTION
double deltaE_cell(const prims& PL,
                   const prims& Pc,
                   const prims& PR,
                   const double dt_dx) noexcept {
    if (Pc.vx >= 0.0) {
        return deltaE_pair(PL, Pc, dt_dx, Pc.vx);
    }
    return deltaE_pair(Pc, PR, dt_dx, Pc.vx);
}

KOKKOS_INLINE_FUNCTION
cons linear_correction_impl(
    const cons&  C_in,
    const prims& P_tgt,
    const double gamma
) noexcept {
    cons C_out = C_in;

    constexpr int    MAX_ITER  = 20;
    constexpr double TOL_P     = 1.0e-12;
    constexpr double TOL_VX    = 1.0e-12;
    constexpr double NU        = 1.0e-8;
    constexpr double DET_TOL   = 1.0e-15;
    constexpr double EPS       = 1.0e-14;
    constexpr double OMEGA_MIN = 1.0e-8;

    auto cons_to_prims_euler =
        [&](const cons& C, prims& P) noexcept -> bool {
            const double rho = C.rho;
            const double mx  = C.mx;
            const double my  = C.my;
            const double mz  = C.mz;
            const double E   = C.E;

            const double vx = mx / rho;
            const double vy = my / rho;
            const double vz = mz / rho;
            const double kinetic = 0.5 * (mx*mx + my*my + mz*mz) / rho;
            const double p = (gamma - 1.0) * (E - kinetic);

            P.rho = rho;
            P.vx  = vx;
            P.vy  = vy;
            P.vz  = vz;
            P.p   = p;
            return true;
        };

    auto eval_state =
        [&](double mx_k, double E_k,
            prims& Pk,
            double& rp,
            double& rvx) noexcept -> bool {

            cons Ck = C_in;
            Ck.mx = mx_k;
            Ck.E  = E_k;
            Ck.rho = C_in.rho;
            Ck.my  = C_in.my;
            Ck.mz  = C_in.mz;



            if (!cons_to_prims_euler(Ck, Pk)) {
                return false;
            }

            rp  = Pk.p  - P_tgt.p;
            rvx = Pk.vx - P_tgt.vx;
            return true;
        };

    if (!(gamma > 1.0) || !(C_in.rho > EPS) || !(P_tgt.p > EPS)) {
        return C_in;
    }

    double mx_k = C_in.mx;
    double E_k  = C_in.E;

    for (int k = 0; k < MAX_ITER; ++k) {
        prims P1{};
        double r_p  = 0.0;
        double r_vx = 0.0;

        if (!eval_state(mx_k, E_k, P1, r_p, r_vx)) {
            return C_out;
        }

        if (std::fabs(r_p) < TOL_P && std::fabs(r_vx) < TOL_VX) {
            break;
        }

        const double eps_mx = NU * fmax(1.0, std::fabs(mx_k));
        const double eps_E  = NU * fmax(1.0, std::fabs(E_k));

        prims P2{}, P3{};
        double rp2 = 0.0, rvx2 = 0.0;

        if (!eval_state(mx_k + eps_mx, E_k, P2, rp2, rvx2)) {
            return C_out;
        }
        const double a = (P2.p  - P1.p ) / eps_mx;
        const double c = (P2.vx - P1.vx) / eps_mx;

        if (!eval_state(mx_k, E_k + eps_E, P3, rp2, rvx2)) {
            return C_out;
        }
        const double b = (P3.p  - P1.p ) / eps_E;
        const double d = (P3.vx - P1.vx) / eps_E;

        const double det = a * d - b * c;
        if (!(std::fabs(det) > DET_TOL) || !std::isfinite(det)) {
            break;
        }

        const double inv_det = 1.0 / det;
        const double dmx = (-d * r_p + b * r_vx) * inv_det;
        const double dE  = ( c * r_p - a * r_vx) * inv_det;

        if (!std::isfinite(dmx) || !std::isfinite(dE)) {
            return C_out;
        }

        double omega = 1.0;
        bool accepted = false;
        const double phi0 = 0.5 * (r_p * r_p + r_vx * r_vx);

        while (omega >= OMEGA_MIN) {
            const double mx_try = mx_k + omega * dmx;
            const double E_try  = E_k  + omega * dE;

            prims P_try{};
            double rp_try  = 0.0;
            double rvx_try = 0.0;

            if (!eval_state(mx_try, E_try, P_try, rp_try, rvx_try)) {
                omega *= 0.5;
                continue;
            }

            const double phi_try = 0.5 * (rp_try * rp_try + rvx_try * rvx_try);
            if (std::isfinite(phi_try) && phi_try < phi0) {
                mx_k = mx_try;
                E_k  = E_try;
                accepted = true;
                break;
            }

            omega *= 0.5;
        }

        if (!accepted) {
            break;
        }
    }

    C_out.rho = C_in.rho;
    C_out.mx  = mx_k;
    C_out.my  = C_in.my;
    C_out.mz  = C_in.mz;
    C_out.E   = E_k;

    return C_out;
}

KOKKOS_INLINE_FUNCTION
cons linear_correction(const cons&  C_in,
                       const prims& PL,
                       const prims& Pc,
                       const prims& PR,
                       const double gamma,
                        aether::core::Simulation::View sim,
                        int k, int j, int i) noexcept {

    auto p_new = cons_to_prims_cell(C_in,sim.gamma);

    if ( (p_new.p - fmax( PL.p, fmax(PR.p,Pc.p) )) > 1.e-7 ){
        
        sim.contact_wave(k,j,i) = 1.0;
        return linear_correction_impl(C_in, Pc, gamma);
    } else{
        sim.contact_wave(k,j,i) = 0.0;
        return C_in;
    }
    sim.contact_wave(k,j,i) = 1.0;
    return linear_correction_impl(C_in, Pc, gamma);
}

} // namespace aether::physics::euler

namespace aether::core {
    void correct_domain(Simulation& sim);
}
