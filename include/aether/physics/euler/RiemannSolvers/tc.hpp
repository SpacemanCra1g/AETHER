#pragma once
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/euler/convert.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>

namespace aether::physics::euler {

KOKKOS_INLINE_FUNCTION
prims tc(const prims& L, const prims& R, double gamma, double /*dt_dx*/) noexcept {
    prims Flux{};

    const double eps     = 1.0e-14;
    const double gm1     = gamma - 1.0;
    const double inv_gm1 = 1.0 / gm1;

    auto finite4 = [](double a, double b, double c, double d) noexcept {
        return std::isfinite(a) && std::isfinite(b) &&
               std::isfinite(c) && std::isfinite(d);
    };

    auto physical_flux = [&](const prims& Q) noexcept -> prims {
        prims F{};
        const double v2 = Q.vx*Q.vx + Q.vy*Q.vy + Q.vz*Q.vz;
        const double E  = Q.p * inv_gm1 + 0.5 * Q.rho * v2;

        F.rho = Q.rho * Q.vx;
        F.vx  = Q.rho * Q.vx * Q.vx + Q.p;
        F.vy  = Q.rho * Q.vx * Q.vy;
        F.vz  = 0.0;
        F.p   = Q.vx * (E + Q.p);
        return F;
    };

    if (!(gamma > 1.0) ||
        !(L.rho > 0.0) || !(R.rho > 0.0) ||
        !(L.p   > 0.0) || !(R.p   > 0.0)) {
        return physical_flux(L);
    }

    // Contact-wave experimental flux:
    // Build a single interface contact state and compute all fluxes from it,
    // so that transverse momentum and total energy are transported consistently.

    // Contact speed and pressure are forced flat.
    const double u_star = 0.5 * (L.vx + R.vx);
    const double p_star = 0.5 * (L.p  + R.p);

    // Upwind the contact-carried quantities using the contact speed.
    // If u_star ~ 0, use a simple average.
    double rho_star, vy_star;
    if (u_star > eps) {
        rho_star = L.rho;
        vy_star  = L.vy;
    } else if (u_star < -eps) {
        rho_star = R.rho;
        vy_star  = R.vy;
    } else {
        rho_star = 0.5 * (L.rho + R.rho);
        vy_star  = 0.5 * (L.vy  + R.vy);
    }

    // Full contact-state total energy, including transverse kinetic energy.
    const double E_star =
        p_star * inv_gm1 +
        0.5 * rho_star * (u_star*u_star + vy_star*vy_star);

    // Fluxes computed consistently from the same contact state.
    const double F_rho = rho_star * u_star;
    const double F_mx  = rho_star * u_star * u_star + p_star;
    const double F_my  = rho_star * u_star * vy_star;
    const double F_E   = u_star * (E_star + p_star);

    if (!finite4(F_rho, F_mx, F_my, F_E)) {
        return physical_flux(L);
    }

    Flux.rho = F_rho;
    Flux.vx  = F_mx;
    Flux.vy  = F_my;
    Flux.vz  = 0.0;
    Flux.p   = F_E;

    return Flux;
}

} // namespace aether::physics::euler