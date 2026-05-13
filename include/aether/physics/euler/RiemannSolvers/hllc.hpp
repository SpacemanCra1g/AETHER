#pragma once
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/euler/convert.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>

namespace aether::physics::euler {

KOKKOS_INLINE_FUNCTION
prims hllc(const prims& L, const prims& R, double gamma) noexcept {
    prims Flux{};

    const double inv_gm1 = 1.0 / (gamma - 1.0);

    // -----------------------------
    // Left/right primitive helpers
    // -----------------------------
    const double rhoL = L.rho;
    const double uL   = L.vx;
    const double vL   = L.vy;
    const double wL   = L.vz;
    const double pL   = L.p;

    const double rhoR = R.rho;
    const double uR   = R.vx;
    const double vR   = R.vy;
    const double wR   = R.vz;
    const double pR   = R.p;

    const double aL = sqrt(gamma * pL / rhoL);
    const double aR = sqrt(gamma * pR / rhoR);

    // Total energies
    const double v2L = uL*uL + vL*vL + wL*wL;
    const double v2R = uR*uR + vR*vR + wR*wR;

    const double EL = pL * inv_gm1 + 0.5 * rhoL * v2L;
    const double ER = pR * inv_gm1 + 0.5 * rhoR * v2R;

    // Conservative states
    const double UL0 = rhoL;
    const double UL1 = rhoL * uL;
    const double UL2 = rhoL * vL;
    const double UL3 = rhoL * wL;
    const double UL4 = EL;

    const double UR0 = rhoR;
    const double UR1 = rhoR * uR;
    const double UR2 = rhoR * vR;
    const double UR3 = rhoR * wR;
    const double UR4 = ER;

    // Physical fluxes
    const double FL0 = UL1;
    const double FL1 = rhoL * uL * uL + pL;
    const double FL2 = rhoL * uL * vL;
    const double FL3 = rhoL * uL * wL;
    const double FL4 = uL * (EL + pL);

    const double FR0 = UR1;
    const double FR1 = rhoR * uR * uR + pR;
    const double FR2 = rhoR * uR * vR;
    const double FR3 = rhoR * uR * wR;
    const double FR4 = uR * (ER + pR);

    // -----------------------------
    // Signal-speed estimates
    // -----------------------------
    const double SL = fmin(uL - aL, uR - aR);
    const double SR = fmax(uL + aL, uR + aR);

    // Supersonic left
    if (0.0 <= SL) {
        Flux = flux_from_prim_cell(L, gamma);
    }

    // Supersonic right
    else if (SR <= 0.0) {
        Flux = flux_from_prim_cell(R, gamma);
    }
    else {

        // -----------------------------
        // Contact wave speed S*
        // -----------------------------
        const double num = pR - pL
                         + rhoL * uL * (SL - uL)
                         - rhoR * uR * (SR - uR);

        const double den = rhoL * (SL - uL)
                         - rhoR * (SR - uR);

        const double SM = num / den;

        // Star-region pressure from either side
        const double pStarL = pL + rhoL * (SL - uL) * (SM - uL);
        const double pStarR = pR + rhoR * (SR - uR) * (SM - uR);
        const double pStar  = 0.5 * (pStarL + pStarR);

        // -----------------------------
        // Left star state
        // -----------------------------
        const double rhoStarL = rhoL * (SL - uL) / (SL - SM);

        const double UStarL0 = rhoStarL;
        const double UStarL1 = rhoStarL * SM;
        const double UStarL2 = rhoStarL * vL;
        const double UStarL3 = rhoStarL * wL;

        const double EStarL =
            ((SL - uL) * EL - pL * uL + pStar * SM) / (SL - SM);

        const double UStarL4 = EStarL;

        // -----------------------------
        // Right star state
        // -----------------------------
        const double rhoStarR = rhoR * (SR - uR) / (SR - SM);

        const double UStarR0 = rhoStarR;
        const double UStarR1 = rhoStarR * SM;
        const double UStarR2 = rhoStarR * vR;
        const double UStarR3 = rhoStarR * wR;

        const double EStarR =
            ((SR - uR) * ER - pR * uR + pStar * SM) / (SR - SM);

        const double UStarR4 = EStarR;

        // -----------------------------
        // HLLC flux selection
        // -----------------------------
        if (0.0 <= SM) {
            Flux.rho = FL0 + SL * (UStarL0 - UL0);
            Flux.vx  = FL1 + SL * (UStarL1 - UL1);
            Flux.vy  = FL2 + SL * (UStarL2 - UL2);
            Flux.vz  = FL3 + SL * (UStarL3 - UL3);
            Flux.p   = FL4 + SL * (UStarL4 - UL4);
        } else {
            Flux.rho = FR0 + SR * (UStarR0 - UR0);
            Flux.vx  = FR1 + SR * (UStarR1 - UR1);
            Flux.vy  = FR2 + SR * (UStarR2 - UR2);
            Flux.vz  = FR3 + SR * (UStarR3 - UR3);
            Flux.p   = FR4 + SR * (UStarR4 - UR4);
        }
    }


    // Calculate and store the internal_energy flux and int_e source flux
    Flux.e = Flux.rho * ( (Flux.rho >= 0.0)
                       ? L.e / L.rho
                       : R.e / R.rho );

    return Flux;
}
}
