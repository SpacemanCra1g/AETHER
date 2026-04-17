#pragma once
#include <Kokkos_Macros.hpp>
#include "aether/core/config_build.hpp"
#include <aether/math/mats.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <aether/core/views.hpp>
#include "aether/core/char_struct.hpp"
#include "aether/core/prim_layout.hpp"
#include <aether/core/simulation.hpp>

namespace aether::physics::euler {

using P = aether::prim::Prim;

KOKKOS_INLINE_FUNCTION
static void D1_X_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.x.right(0,0) = 1.0;
    chars.x.right(0,1) = inv_a2;
    chars.x.right(0,2) = inv_a2;

    chars.x.right(1,0) = 0.0;
    chars.x.right(1,1) = -inv_a * inv_rho;
    chars.x.right(1,2) =  inv_a * inv_rho;

    chars.x.right(2,0) = 0.0;
    chars.x.right(2,1) = 1.0;
    chars.x.right(2,2) = 1.0;

    chars.x.left(0,0) = 1.0;
    chars.x.left(0,1) = 0.0;
    chars.x.left(0,2) = -inv_a2;

    chars.x.left(1,0) = 0.0;
    chars.x.left(1,1) = -0.5 * a * p.rho;
    chars.x.left(1,2) =  0.5;

    chars.x.left(2,0) = 0.0;
    chars.x.left(2,1) =  0.5 * a * p.rho;
    chars.x.left(2,2) =  0.5;

    chars.x.eigs[0] = p.vx;
    chars.x.eigs[1] = p.vx - a;
    chars.x.eigs[2] = p.vx + a;
}

KOKKOS_INLINE_FUNCTION
static void D2_X_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.x.right(0,0) = 1.0;
    chars.x.right(0,1) = 0.0;
    chars.x.right(0,2) = inv_a2;
    chars.x.right(0,3) = inv_a2;

    chars.x.right(1,0) = 0.0;
    chars.x.right(1,1) = 0.0;
    chars.x.right(1,2) = -inv_a * inv_rho;
    chars.x.right(1,3) =  inv_a * inv_rho;

    chars.x.right(2,0) = 0.0;
    chars.x.right(2,1) = 1.0;
    chars.x.right(2,2) = 0.0;
    chars.x.right(2,3) = 0.0;

    chars.x.right(3,0) = 0.0;
    chars.x.right(3,1) = 0.0;
    chars.x.right(3,2) = 1.0;
    chars.x.right(3,3) = 1.0;

    chars.x.left(0,0) = 1.0;
    chars.x.left(0,1) = 0.0;
    chars.x.left(0,2) = 0.0;
    chars.x.left(0,3) = -inv_a2;

    chars.x.left(1,0) = 0.0;
    chars.x.left(1,1) = 0.0;
    chars.x.left(1,2) = 1.0;
    chars.x.left(1,3) = 0.0;

    chars.x.left(2,0) = 0.0;
    chars.x.left(2,1) = -0.5 * a * p.rho;
    chars.x.left(2,2) = 0.0;
    chars.x.left(2,3) = 0.5;

    chars.x.left(3,0) = 0.0;
    chars.x.left(3,1) =  0.5 * a * p.rho;
    chars.x.left(3,2) = 0.0;
    chars.x.left(3,3) = 0.5;

    chars.x.eigs[0] = p.vx;
    chars.x.eigs[1] = p.vx;
    chars.x.eigs[2] = p.vx - a;
    chars.x.eigs[3] = p.vx + a;
}

KOKKOS_INLINE_FUNCTION
static void D3_X_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.x.right(0,0) = 1.0;
    chars.x.right(0,1) = 0.0;
    chars.x.right(0,2) = 0.0;
    chars.x.right(0,3) = inv_a2;
    chars.x.right(0,4) = inv_a2;

    chars.x.right(1,0) = 0.0;
    chars.x.right(1,1) = 0.0;
    chars.x.right(1,2) = 0.0;
    chars.x.right(1,3) = -inv_a * inv_rho;
    chars.x.right(1,4) =  inv_a * inv_rho;

    chars.x.right(2,0) = 0.0;
    chars.x.right(2,1) = 1.0;
    chars.x.right(2,2) = 0.0;
    chars.x.right(2,3) = 0.0;
    chars.x.right(2,4) = 0.0;

    chars.x.right(3,0) = 0.0;
    chars.x.right(3,1) = 0.0;
    chars.x.right(3,2) = 1.0;
    chars.x.right(3,3) = 0.0;
    chars.x.right(3,4) = 0.0;

    chars.x.right(4,0) = 0.0;
    chars.x.right(4,1) = 0.0;
    chars.x.right(4,2) = 0.0;
    chars.x.right(4,3) = 1.0;
    chars.x.right(4,4) = 1.0;

    chars.x.left(0,0) = 1.0;
    chars.x.left(0,1) = 0.0;
    chars.x.left(0,2) = 0.0;
    chars.x.left(0,3) = 0.0;
    chars.x.left(0,4) = -inv_a2;

    chars.x.left(1,0) = 0.0;
    chars.x.left(1,1) = 0.0;
    chars.x.left(1,2) = 1.0;
    chars.x.left(1,3) = 0.0;
    chars.x.left(1,4) = 0.0;

    chars.x.left(2,0) = 0.0;
    chars.x.left(2,1) = 0.0;
    chars.x.left(2,2) = 0.0;
    chars.x.left(2,3) = 1.0;
    chars.x.left(2,4) = 0.0;

    chars.x.left(3,0) = 0.0;
    chars.x.left(3,1) = -0.5 * a * p.rho;
    chars.x.left(3,2) = 0.0;
    chars.x.left(3,3) = 0.0;
    chars.x.left(3,4) = 0.5;

    chars.x.left(4,0) = 0.0;
    chars.x.left(4,1) =  0.5 * a * p.rho;
    chars.x.left(4,2) = 0.0;
    chars.x.left(4,3) = 0.0;
    chars.x.left(4,4) = 0.5;

    chars.x.eigs[0] = p.vx;
    chars.x.eigs[1] = p.vx;
    chars.x.eigs[2] = p.vx;
    chars.x.eigs[3] = p.vx - a;
    chars.x.eigs[4] = p.vx + a;
}

KOKKOS_INLINE_FUNCTION
static void D2_Y_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.y.right(0,0) = 1.0;
    chars.y.right(0,1) = 0.0;
    chars.y.right(0,2) = inv_a2;
    chars.y.right(0,3) = inv_a2;

    chars.y.right(1,0) = 0.0;
    chars.y.right(1,1) = 1.0;
    chars.y.right(1,2) = 0.0;
    chars.y.right(1,3) = 0.0;

    chars.y.right(2,0) = 0.0;
    chars.y.right(2,1) = 0.0;
    chars.y.right(2,2) = -inv_a * inv_rho;
    chars.y.right(2,3) =  inv_a * inv_rho;

    chars.y.right(3,0) = 0.0;
    chars.y.right(3,1) = 0.0;
    chars.y.right(3,2) = 1.0;
    chars.y.right(3,3) = 1.0;

    chars.y.left(0,0) = 1.0;
    chars.y.left(0,1) = 0.0;
    chars.y.left(0,2) = 0.0;
    chars.y.left(0,3) = -inv_a2;

    chars.y.left(1,0) = 0.0;
    chars.y.left(1,1) = 1.0;
    chars.y.left(1,2) = 0.0;
    chars.y.left(1,3) = 0.0;

    chars.y.left(2,0) = 0.0;
    chars.y.left(2,1) = 0.0;
    chars.y.left(2,2) = -0.5 * a * p.rho;
    chars.y.left(2,3) = 0.5;

    chars.y.left(3,0) = 0.0;
    chars.y.left(3,1) = 0.0;
    chars.y.left(3,2) =  0.5 * a * p.rho;
    chars.y.left(3,3) = 0.5;

    chars.y.eigs[0] = p.vy;
    chars.y.eigs[1] = p.vy;
    chars.y.eigs[2] = p.vy - a;
    chars.y.eigs[3] = p.vy + a;
}

KOKKOS_INLINE_FUNCTION
static void D3_Y_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.y.right(0,0) = 1.0;
    chars.y.right(0,1) = 0.0;
    chars.y.right(0,2) = 0.0;
    chars.y.right(0,3) = inv_a2;
    chars.y.right(0,4) = inv_a2;

    chars.y.right(1,0) = 0.0;
    chars.y.right(1,1) = 1.0;
    chars.y.right(1,2) = 0.0;
    chars.y.right(1,3) = 0.0;
    chars.y.right(1,4) = 0.0;

    chars.y.right(2,0) = 0.0;
    chars.y.right(2,1) = 0.0;
    chars.y.right(2,2) = 0.0;
    chars.y.right(2,3) = -inv_a * inv_rho;
    chars.y.right(2,4) =  inv_a * inv_rho;

    chars.y.right(3,0) = 0.0;
    chars.y.right(3,1) = 0.0;
    chars.y.right(3,2) = 1.0;
    chars.y.right(3,3) = 0.0;
    chars.y.right(3,4) = 0.0;

    chars.y.right(4,0) = 0.0;
    chars.y.right(4,1) = 0.0;
    chars.y.right(4,2) = 0.0;
    chars.y.right(4,3) = 1.0;
    chars.y.right(4,4) = 1.0;

    chars.y.left(0,0) = 1.0;
    chars.y.left(0,1) = 0.0;
    chars.y.left(0,2) = 0.0;
    chars.y.left(0,3) = 0.0;
    chars.y.left(0,4) = -inv_a2;

    chars.y.left(1,0) = 0.0;
    chars.y.left(1,1) = 1.0;
    chars.y.left(1,2) = 0.0;
    chars.y.left(1,3) = 0.0;
    chars.y.left(1,4) = 0.0;

    chars.y.left(2,0) = 0.0;
    chars.y.left(2,1) = 0.0;
    chars.y.left(2,2) = 0.0;
    chars.y.left(2,3) = 1.0;
    chars.y.left(2,4) = 0.0;

    chars.y.left(3,0) = 0.0;
    chars.y.left(3,1) = 0.0;
    chars.y.left(3,2) = -0.5 * a * p.rho;
    chars.y.left(3,3) = 0.0;
    chars.y.left(3,4) = 0.5;

    chars.y.left(4,0) = 0.0;
    chars.y.left(4,1) = 0.0;
    chars.y.left(4,2) =  0.5 * a * p.rho;
    chars.y.left(4,3) = 0.0;
    chars.y.left(4,4) = 0.5;

    chars.y.eigs[0] = p.vy;
    chars.y.eigs[1] = p.vy;
    chars.y.eigs[2] = p.vy;
    chars.y.eigs[3] = p.vy - a;
    chars.y.eigs[4] = p.vy + a;
}

KOKKOS_INLINE_FUNCTION
static void D3_Z_EigenMatrix(
    const prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2      = p.p * gamma / p.rho;
    const double inv_a2  = 1.0 / a2;
    const double a       = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a   = 1.0 / a;

    chars.z.right(0,0) = 1.0;
    chars.z.right(0,1) = 0.0;
    chars.z.right(0,2) = 0.0;
    chars.z.right(0,3) = inv_a2;
    chars.z.right(0,4) = inv_a2;

    chars.z.right(1,0) = 0.0;
    chars.z.right(1,1) = 1.0;
    chars.z.right(1,2) = 0.0;
    chars.z.right(1,3) = 0.0;
    chars.z.right(1,4) = 0.0;

    chars.z.right(2,0) = 0.0;
    chars.z.right(2,1) = 0.0;
    chars.z.right(2,2) = 1.0;
    chars.z.right(2,3) = 0.0;
    chars.z.right(2,4) = 0.0;

    chars.z.right(3,0) = 0.0;
    chars.z.right(3,1) = 0.0;
    chars.z.right(3,2) = 0.0;
    chars.z.right(3,3) = -inv_a * inv_rho;
    chars.z.right(3,4) =  inv_a * inv_rho;

    chars.z.right(4,0) = 0.0;
    chars.z.right(4,1) = 0.0;
    chars.z.right(4,2) = 0.0;
    chars.z.right(4,3) = 1.0;
    chars.z.right(4,4) = 1.0;

    chars.z.left(0,0) = 1.0;
    chars.z.left(0,1) = 0.0;
    chars.z.left(0,2) = 0.0;
    chars.z.left(0,3) = 0.0;
    chars.z.left(0,4) = -inv_a2;

    chars.z.left(1,0) = 0.0;
    chars.z.left(1,1) = 1.0;
    chars.z.left(1,2) = 0.0;
    chars.z.left(1,3) = 0.0;
    chars.z.left(1,4) = 0.0;

    chars.z.left(2,0) = 0.0;
    chars.z.left(2,1) = 0.0;
    chars.z.left(2,2) = 1.0;
    chars.z.left(2,3) = 0.0;
    chars.z.left(2,4) = 0.0;

    chars.z.left(3,0) = 0.0;
    chars.z.left(3,1) = 0.0;
    chars.z.left(3,2) = 0.0;
    chars.z.left(3,3) = -0.5 * a * p.rho;
    chars.z.left(3,4) = 0.5;

    chars.z.left(4,0) = 0.0;
    chars.z.left(4,1) = 0.0;
    chars.z.left(4,2) = 0.0;
    chars.z.left(4,3) =  0.5 * a * p.rho;
    chars.z.left(4,4) = 0.5;

    chars.z.eigs[0] = p.vz;
    chars.z.eigs[1] = p.vz;
    chars.z.eigs[2] = p.vz;
    chars.z.eigs[3] = p.vz - a;
    chars.z.eigs[4] = p.vz + a;
}

KOKKOS_INLINE_FUNCTION
void fill_eigenvectors(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    if constexpr (AETHER_DIM == 1) {
        if constexpr (P::COUNT == 3) {
            D1_X_EigenMatrix(p, chars, gamma);
        } else if constexpr (P::COUNT == 4) {
            D2_X_EigenMatrix(p, chars, gamma);
        } else {
            D3_X_EigenMatrix(p, chars, gamma);
        }
    }
    else if constexpr (AETHER_DIM == 2) {
        if constexpr (P::COUNT == 4) {
            D2_X_EigenMatrix(p, chars, gamma);
            D2_Y_EigenMatrix(p, chars, gamma);
        } else {
            D3_X_EigenMatrix(p, chars, gamma);
            D3_Y_EigenMatrix(p, chars, gamma);
        }
    }
    else if constexpr (AETHER_DIM == 3) {
        D3_X_EigenMatrix(p, chars, gamma);
        D3_Y_EigenMatrix(p, chars, gamma);
        D3_Z_EigenMatrix(p, chars, gamma);
    }
}

} // namespace aether::physics::euler