#pragma once

#include "aether/core/config.hpp"
#include <aether/math/mats.hpp>
#include <aether/physics/counts.hpp>
#include <Kokkos_Macros.hpp>

namespace aether::core {

template <int N>
using eigvals_t = aether::math::Vec<N>;

template <int N>
struct spectral_dir_data {
    aether::math::Mat<N> left {};
    aether::math::Mat<N> right {};
    eigvals_t<N> eigs {};

    KOKKOS_INLINE_FUNCTION
    double& lambda(const int m) noexcept {
        return eigs[m];
    }

    KOKKOS_INLINE_FUNCTION
    const double& lambda(const int m) const noexcept {
        return eigs[m];
    }
};

template <int Dim, int NumVar>
struct spectral_cell_t {
    spectral_dir_data<NumVar> x {};
    spectral_dir_data<NumVar> y {};
    spectral_dir_data<NumVar> z {};

    KOKKOS_INLINE_FUNCTION
    spectral_dir_data<NumVar>& x_dir() noexcept {
        return x;
    }

    KOKKOS_INLINE_FUNCTION
    const spectral_dir_data<NumVar>& x_dir() const noexcept {
        return x;
    }

    KOKKOS_INLINE_FUNCTION
    spectral_dir_data<NumVar>& y_dir() noexcept {
        static_assert(Dim > 1, "y_dir() is only valid when Dim > 1");
        return y;
    }

    KOKKOS_INLINE_FUNCTION
    const spectral_dir_data<NumVar>& y_dir() const noexcept {
        static_assert(Dim > 1, "y_dir() is only valid when Dim > 1");
        return y;
    }

    KOKKOS_INLINE_FUNCTION
    spectral_dir_data<NumVar>& z_dir() noexcept {
        static_assert(Dim > 2, "z_dir() is only valid when Dim > 2");
        return z;
    }

    KOKKOS_INLINE_FUNCTION
    const spectral_dir_data<NumVar>& z_dir() const noexcept {
        static_assert(Dim > 2, "z_dir() is only valid when Dim > 2");
        return z;
    }
};

template <int Dim, int NumVar>
using one_cell_spectral_containerT = spectral_cell_t<Dim, NumVar>;

using one_cell_spectral_container =
    one_cell_spectral_containerT<AETHER_DIM, aether::phys_ct::numvar-1>;

} // namespace aether::core
