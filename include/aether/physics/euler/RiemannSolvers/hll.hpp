#pragma once 
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <cmath>

namespace aether::physics::euler{
AETHER_INLINE prims hll(const prims& L, const prims& R, const double gamma) noexcept {
    prims Flux{};

    const double inv_gm1 = 1.0 / (gamma - 1.0);

    // sound speeds
    const double aL = std::sqrt(gamma * L.p / L.rho);
    const double aR = std::sqrt(gamma * R.p / R.rho);

    // wave speed estimates
    const double SL = std::fmin(L.vx - aL, R.vx - aR);
    const double SR = std::fmax(L.vx + aL, R.vx + aR);

    // Helper lambda, New syntax to me WOOT! 
    auto flux_from_prim = [&](const prims& W) -> prims {
        prims F{};
        const double v2 = W.vx*W.vx + W.vy*W.vy + W.vz*W.vz;
        const double E  = W.p * inv_gm1 + 0.5 * W.rho * v2;

        F.rho = W.rho * W.vx;
        F.vx  = W.rho * W.vx * W.vx + W.p;
        F.vy  = W.rho * W.vx * W.vy;
        F.vz  = W.rho * W.vx * W.vz;
        F.p   = W.vx * (E + W.p); 
        return F;
    };

    // Supersonic left
    if (0.0 <= SL) {
        return flux_from_prim(L);
    }

    // Supersonic right
    if (SR <= 0.0) {
        return flux_from_prim(R);
    }

    // Star region (HLL)
    const double v2L = L.vx*L.vx + L.vy*L.vy + L.vz*L.vz;
    const double EL  = L.p * inv_gm1 + 0.5 * L.rho * v2L;

    const double v2R = R.vx*R.vx + R.vy*R.vy + R.vz*R.vz;
    const double ER  = R.p * inv_gm1 + 0.5 * R.rho * v2R;

    // Conservative states (Euler)
    const double UL0 = L.rho;
    const double UL1 = L.rho * L.vx;
    const double UL2 = L.rho * L.vy;
    const double UL3 = L.rho * L.vz;
    const double UL4 = EL;

    const double UR0 = R.rho;
    const double UR1 = R.rho * R.vx;
    const double UR2 = R.rho * R.vy;
    const double UR3 = R.rho * R.vz;
    const double UR4 = ER;

    // Physical fluxes
    const double FL0 = UL1;
    const double FL1 = L.rho * L.vx * L.vx + L.p;
    const double FL2 = L.rho * L.vx * L.vy;
    const double FL3 = L.rho * L.vx * L.vz;
    const double FL4 = L.vx * (EL + L.p);

    const double FR0 = UR1;
    const double FR1 = R.rho * R.vx * R.vx + R.p;
    const double FR2 = R.rho * R.vx * R.vy;
    const double FR3 = R.rho * R.vx * R.vz;
    const double FR4 = R.vx * (ER + R.p);

    const double inv_dS = 1.0 / (SR - SL);
    const double SRSL   = SR * SL;

    Flux.rho = inv_dS * (SR*FL0 - SL*FR0 + SRSL*(UR0 - UL0));
    Flux.vx  = inv_dS * (SR*FL1 - SL*FR1 + SRSL*(UR1 - UL1));
    Flux.vy  = inv_dS * (SR*FL2 - SL*FR2 + SRSL*(UR2 - UL2));
    Flux.vz  = inv_dS * (SR*FL3 - SL*FR3 + SRSL*(UR3 - UL3));
    Flux.p   = inv_dS * (SR*FL4 - SL*FR4 + SRSL*(UR4 - UL4));

    return Flux;
}
}