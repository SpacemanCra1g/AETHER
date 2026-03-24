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
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.x_right)(0,0) = 1.0;
    (*chars.x_right)(0,1) = inv_a2;
    (*chars.x_right)(0,2) = inv_a2;

    (*chars.x_right)(1,0) = 0.0;
    (*chars.x_right)(1,1) = -inv_a * inv_rho;
    (*chars.x_right)(1,2) =  inv_a * inv_rho;

    (*chars.x_right)(2,0) = 0.0;
    (*chars.x_right)(2,1) = 1.0;
    (*chars.x_right)(2,2) = 1.0;

    (*chars.x_left)(0,0) = 1.0;
    (*chars.x_left)(0,1) = 0.0;
    (*chars.x_left)(0,2) = -inv_a2;

    (*chars.x_left)(1,0) = 0.0;
    (*chars.x_left)(1,1) = -0.5 * a * p.rho;
    (*chars.x_left)(1,2) =  0.5;

    (*chars.x_left)(2,0) = 0.0;
    (*chars.x_left)(2,1) =  0.5 * a * p.rho;
    (*chars.x_left)(2,2) =  0.5;

    (*chars.x_eigs) = {p.vx, p.vx - a, p.vx + a, 0.0, 0.0};
}

KOKKOS_INLINE_FUNCTION
static void D2_X_EigenMatrix(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.x_right)(0,0) = 1.0;
    (*chars.x_right)(0,1) = 0.0;
    (*chars.x_right)(0,2) = inv_a2;
    (*chars.x_right)(0,3) = inv_a2;

    (*chars.x_right)(1,0) = 0.0;
    (*chars.x_right)(1,1) = 0.0;
    (*chars.x_right)(1,2) = -inv_a * inv_rho;
    (*chars.x_right)(1,3) =  inv_a * inv_rho;

    (*chars.x_right)(2,0) = 0.0;
    (*chars.x_right)(2,1) = 1.0;
    (*chars.x_right)(2,2) = 0.0;
    (*chars.x_right)(2,3) = 0.0;

    (*chars.x_right)(3,0) = 0.0;
    (*chars.x_right)(3,1) = 0.0;
    (*chars.x_right)(3,2) = 1.0;
    (*chars.x_right)(3,3) = 1.0;

    (*chars.x_left)(0,0) = 1.0;
    (*chars.x_left)(0,1) = 0.0;
    (*chars.x_left)(0,2) = 0.0;
    (*chars.x_left)(0,3) = -inv_a2;

    (*chars.x_left)(1,0) = 0.0;
    (*chars.x_left)(1,1) = 0.0;
    (*chars.x_left)(1,2) = 1.0;
    (*chars.x_left)(1,3) = 0.0;

    (*chars.x_left)(2,0) = 0.0;
    (*chars.x_left)(2,1) = -0.5 * a * p.rho;
    (*chars.x_left)(2,2) = 0.0;
    (*chars.x_left)(2,3) = 0.5;

    (*chars.x_left)(3,0) = 0.0;
    (*chars.x_left)(3,1) =  0.5 * a * p.rho;
    (*chars.x_left)(3,2) = 0.0;
    (*chars.x_left)(3,3) = 0.5;

    (*chars.x_eigs) = {p.vx, p.vx, p.vx - a, p.vx + a, 0.0};
}

KOKKOS_INLINE_FUNCTION
static void D3_X_EigenMatrix(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.x_right)(0,0) = 1.0;
    (*chars.x_right)(0,1) = 0.0;
    (*chars.x_right)(0,2) = 0.0;
    (*chars.x_right)(0,3) = inv_a2;
    (*chars.x_right)(0,4) = inv_a2;

    (*chars.x_right)(1,0) = 0.0;
    (*chars.x_right)(1,1) = 0.0;
    (*chars.x_right)(1,2) = 0.0;
    (*chars.x_right)(1,3) = -inv_a * inv_rho;
    (*chars.x_right)(1,4) =  inv_a * inv_rho;

    (*chars.x_right)(2,0) = 0.0;
    (*chars.x_right)(2,1) = 1.0;
    (*chars.x_right)(2,2) = 0.0;
    (*chars.x_right)(2,3) = 0.0;
    (*chars.x_right)(2,4) = 0.0;

    (*chars.x_right)(3,0) = 0.0;
    (*chars.x_right)(3,1) = 0.0;
    (*chars.x_right)(3,2) = 1.0;
    (*chars.x_right)(3,3) = 0.0;
    (*chars.x_right)(3,4) = 0.0;

    (*chars.x_right)(4,0) = 0.0;
    (*chars.x_right)(4,1) = 0.0;
    (*chars.x_right)(4,2) = 0.0;
    (*chars.x_right)(4,3) = 1.0;
    (*chars.x_right)(4,4) = 1.0;

    (*chars.x_left)(0,0) = 1.0;
    (*chars.x_left)(0,1) = 0.0;
    (*chars.x_left)(0,2) = 0.0;
    (*chars.x_left)(0,3) = 0.0;
    (*chars.x_left)(0,4) = -inv_a2;

    (*chars.x_left)(1,0) = 0.0;
    (*chars.x_left)(1,1) = 0.0;
    (*chars.x_left)(1,2) = 1.0;
    (*chars.x_left)(1,3) = 0.0;
    (*chars.x_left)(1,4) = 0.0;

    (*chars.x_left)(2,0) = 0.0;
    (*chars.x_left)(2,1) = 0.0;
    (*chars.x_left)(2,2) = 0.0;
    (*chars.x_left)(2,3) = 1.0;
    (*chars.x_left)(2,4) = 0.0;

    (*chars.x_left)(3,0) = 0.0;
    (*chars.x_left)(3,1) = -0.5 * a * p.rho;
    (*chars.x_left)(3,2) = 0.0;
    (*chars.x_left)(3,3) = 0.0;
    (*chars.x_left)(3,4) = 0.5;

    (*chars.x_left)(4,0) = 0.0;
    (*chars.x_left)(4,1) =  0.5 * a * p.rho;
    (*chars.x_left)(4,2) = 0.0;
    (*chars.x_left)(4,3) = 0.0;
    (*chars.x_left)(4,4) = 0.5;

    (*chars.x_eigs) = {p.vx, p.vx, p.vx, p.vx - a, p.vx + a};
}

KOKKOS_INLINE_FUNCTION
static void D2_Y_EigenMatrix(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.y_right)(0,0) = 1.0;
    (*chars.y_right)(0,1) = 0.0;
    (*chars.y_right)(0,2) = inv_a2;
    (*chars.y_right)(0,3) = inv_a2;

    (*chars.y_right)(1,0) = 0.0;
    (*chars.y_right)(1,1) = 1.0;
    (*chars.y_right)(1,2) = 0.0;
    (*chars.y_right)(1,3) = 0.0;

    (*chars.y_right)(2,0) = 0.0;
    (*chars.y_right)(2,1) = 0.0;
    (*chars.y_right)(2,2) = -inv_a * inv_rho;
    (*chars.y_right)(2,3) =  inv_a * inv_rho;

    (*chars.y_right)(3,0) = 0.0;
    (*chars.y_right)(3,1) = 0.0;
    (*chars.y_right)(3,2) = 1.0;
    (*chars.y_right)(3,3) = 1.0;

    (*chars.y_left)(0,0) = 1.0;
    (*chars.y_left)(0,1) = 0.0;
    (*chars.y_left)(0,2) = 0.0;
    (*chars.y_left)(0,3) = -inv_a2;

    (*chars.y_left)(1,0) = 0.0;
    (*chars.y_left)(1,1) = 1.0;
    (*chars.y_left)(1,2) = 0.0;
    (*chars.y_left)(1,3) = 0.0;

    (*chars.y_left)(2,0) = 0.0;
    (*chars.y_left)(2,1) = 0.0;
    (*chars.y_left)(2,2) = -0.5 * a * p.rho;
    (*chars.y_left)(2,3) = 0.5;

    (*chars.y_left)(3,0) = 0.0;
    (*chars.y_left)(3,1) = 0.0;
    (*chars.y_left)(3,2) =  0.5 * a * p.rho;
    (*chars.y_left)(3,3) = 0.5;

    (*chars.y_eigs) = {p.vy, p.vy, p.vy - a, p.vy + a, 0.0};
}

KOKKOS_INLINE_FUNCTION
static void D3_Y_EigenMatrix(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.y_right)(0,0) = 1.0;
    (*chars.y_right)(0,1) = 0.0;
    (*chars.y_right)(0,2) = 0.0;
    (*chars.y_right)(0,3) = inv_a2;
    (*chars.y_right)(0,4) = inv_a2;

    (*chars.y_right)(1,0) = 0.0;
    (*chars.y_right)(1,1) = 1.0;
    (*chars.y_right)(1,2) = 0.0;
    (*chars.y_right)(1,3) = 0.0;
    (*chars.y_right)(1,4) = 0.0;

    (*chars.y_right)(2,0) = 0.0;
    (*chars.y_right)(2,1) = 0.0;
    (*chars.y_right)(2,2) = 0.0;
    (*chars.y_right)(2,3) = -inv_a * inv_rho;
    (*chars.y_right)(2,4) =  inv_a * inv_rho;

    (*chars.y_right)(3,0) = 0.0;
    (*chars.y_right)(3,1) = 0.0;
    (*chars.y_right)(3,2) = 1.0;
    (*chars.y_right)(3,3) = 0.0;
    (*chars.y_right)(3,4) = 0.0;

    (*chars.y_right)(4,0) = 0.0;
    (*chars.y_right)(4,1) = 0.0;
    (*chars.y_right)(4,2) = 0.0;
    (*chars.y_right)(4,3) = 1.0;
    (*chars.y_right)(4,4) = 1.0;

    (*chars.y_left)(0,0) = 1.0;
    (*chars.y_left)(0,1) = 0.0;
    (*chars.y_left)(0,2) = 0.0;
    (*chars.y_left)(0,3) = 0.0;
    (*chars.y_left)(0,4) = -inv_a2;

    (*chars.y_left)(1,0) = 0.0;
    (*chars.y_left)(1,1) = 1.0;
    (*chars.y_left)(1,2) = 0.0;
    (*chars.y_left)(1,3) = 0.0;
    (*chars.y_left)(1,4) = 0.0;

    (*chars.y_left)(2,0) = 0.0;
    (*chars.y_left)(2,1) = 0.0;
    (*chars.y_left)(2,2) = 0.0;
    (*chars.y_left)(2,3) = 1.0;
    (*chars.y_left)(2,4) = 0.0;

    (*chars.y_left)(3,0) = 0.0;
    (*chars.y_left)(3,1) = 0.0;
    (*chars.y_left)(3,2) = -0.5 * a * p.rho;
    (*chars.y_left)(3,3) = 0.0;
    (*chars.y_left)(3,4) = 0.5;

    (*chars.y_left)(4,0) = 0.0;
    (*chars.y_left)(4,1) = 0.0;
    (*chars.y_left)(4,2) =  0.5 * a * p.rho;
    (*chars.y_left)(4,3) = 0.0;
    (*chars.y_left)(4,4) = 0.5;

    (*chars.y_eigs) = {p.vy, p.vy, p.vy, p.vy - a, p.vy + a};
}

KOKKOS_INLINE_FUNCTION
static void D3_Z_EigenMatrix(
    prims& p,
    aether::core::one_cell_spectral_container& chars,
    const double gamma)
{
    const double a2 = p.p * gamma / p.rho;
    const double inv_a2 = 1.0 / a2;
    const double a = sqrt(a2);
    const double inv_rho = 1.0 / p.rho;
    const double inv_a = 1.0 / a;

    (*chars.z_right)(0,0) = 1.0;
    (*chars.z_right)(0,1) = 0.0;
    (*chars.z_right)(0,2) = 0.0;
    (*chars.z_right)(0,3) = inv_a2;
    (*chars.z_right)(0,4) = inv_a2;

    (*chars.z_right)(1,0) = 0.0;
    (*chars.z_right)(1,1) = 1.0;
    (*chars.z_right)(1,2) = 0.0;
    (*chars.z_right)(1,3) = 0.0;
    (*chars.z_right)(1,4) = 0.0;

    (*chars.z_right)(2,0) = 0.0;
    (*chars.z_right)(2,1) = 0.0;
    (*chars.z_right)(2,2) = 1.0;
    (*chars.z_right)(2,3) = 0.0;
    (*chars.z_right)(2,4) = 0.0;

    (*chars.z_right)(3,0) = 0.0;
    (*chars.z_right)(3,1) = 0.0;
    (*chars.z_right)(3,2) = 0.0;
    (*chars.z_right)(3,3) = -inv_a * inv_rho;
    (*chars.z_right)(3,4) =  inv_a * inv_rho;

    (*chars.z_right)(4,0) = 0.0;
    (*chars.z_right)(4,1) = 0.0;
    (*chars.z_right)(4,2) = 0.0;
    (*chars.z_right)(4,3) = 1.0;
    (*chars.z_right)(4,4) = 1.0;

    (*chars.z_left)(0,0) = 1.0;
    (*chars.z_left)(0,1) = 0.0;
    (*chars.z_left)(0,2) = 0.0;
    (*chars.z_left)(0,3) = 0.0;
    (*chars.z_left)(0,4) = -inv_a2;

    (*chars.z_left)(1,0) = 0.0;
    (*chars.z_left)(1,1) = 1.0;
    (*chars.z_left)(1,2) = 0.0;
    (*chars.z_left)(1,3) = 0.0;
    (*chars.z_left)(1,4) = 0.0;

    (*chars.z_left)(2,0) = 0.0;
    (*chars.z_left)(2,1) = 0.0;
    (*chars.z_left)(2,2) = 1.0;
    (*chars.z_left)(2,3) = 0.0;
    (*chars.z_left)(2,4) = 0.0;

    (*chars.z_left)(3,0) = 0.0;
    (*chars.z_left)(3,1) = 0.0;
    (*chars.z_left)(3,2) = 0.0;
    (*chars.z_left)(3,3) = -0.5 * a * p.rho;
    (*chars.z_left)(3,4) = 0.5;

    (*chars.z_left)(4,0) = 0.0;
    (*chars.z_left)(4,1) = 0.0;
    (*chars.z_left)(4,2) = 0.0;
    (*chars.z_left)(4,3) =  0.5 * a * p.rho;
    (*chars.z_left)(4,4) = 0.5;

    (*chars.z_eigs) = {p.vz, p.vz, p.vz, p.vz - a, p.vz + a};
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

void calc_eigenvecs(aether::core::CellView prim_view,
                    aether::core::eigenvec_view eigs,
                    const double gamma);

} // namespace aether::physics::euler