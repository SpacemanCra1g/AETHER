#pragma once
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/euler/convert.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>

namespace aether::physics::euler {


KOKKOS_INLINE_FUNCTION
prims hll(const prims& L, const prims& R, double gamma) noexcept {
    prims Flux{};

    const double inv_gm1 = 1.0 / (gamma - 1.0);

    // sound speeds
    const double aL = sqrt(gamma * L.p / L.rho);
    const double aR = sqrt(gamma * R.p / R.rho);

    // wave speed estimates
    const double SL = fmin(L.vx - aL, R.vx - aR);
    const double SR = fmax(L.vx + aL, R.vx + aR);

    // Supersonic left
    if (0.0 <= SL) {
        Flux = flux_from_prim_cell(L, gamma);
    }
    // Supersonic right
    else if (SR <= 0.0) {
        Flux = flux_from_prim_cell(R, gamma);
    } else {

    // Star region (HLL)
    const double v2L = L.vx*L.vx + L.vy*L.vy + L.vz*L.vz;
    const double EL  = L.p * inv_gm1 + 0.5 * L.rho * v2L;

    const double v2R = R.vx*R.vx + R.vy*R.vy + R.vz*R.vz;
    const double ER  = R.p * inv_gm1 + 0.5 * R.rho * v2R;

    // Conservative states
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
    }

    // Calculate and store the internal_energy flux and int_e source flux
    Flux.e = Flux.rho * ( (Flux.rho >= 0.0)
                       ? L.e / L.rho
                       : R.e / R.rho );

    return Flux;
}

} // namespace aether::physics::euler
