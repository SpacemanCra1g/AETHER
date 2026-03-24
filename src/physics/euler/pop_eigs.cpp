#include <Kokkos_Core.hpp>
#include "aether/core/char_struct.hpp"
#include "aether/core/config_build.hpp"
#include <aether/core/prim_layout.hpp>
#include <aether/physics/euler/pop_eigs.hpp>

using P = aether::prim::Prim;
namespace co = aether::core;

namespace aether::physics::euler {

void calc_eigenvecs(co::CellView prim_view,
                    co::eigenvec_view eigs,
                    const double gamma)
{
    using exec_space = typename co::Simulation::policy_type::execution_space;

    const std::size_t Nx = prim_view.extent(3);
    const std::size_t Ny = prim_view.extent(2);

    Kokkos::parallel_for(
        "populate_eigensystems",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {0, 0, 0},
            {
                static_cast<int>(prim_view.extent(1)),
                static_cast<int>(prim_view.extent(2)),
                static_cast<int>(prim_view.extent(3))
            }
        ),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            prims prim{};
            co::one_cell_spectral_container chars{};

            prim.rho = prim_view(P::RHO, k, j, i);
            prim.vx  = prim_view(P::VX,  k, j, i);

            if constexpr (P::HAS_VY) {
                prim.vy = prim_view(P::VY, k, j, i);
            } else {
                prim.vy = 0.0;
            }

            if constexpr (P::HAS_VZ) {
                prim.vz = prim_view(P::VZ, k, j, i);
            } else {
                prim.vz = 0.0;
            }

            prim.p = prim_view(P::P, k, j, i);

            const std::size_t idx =
                static_cast<std::size_t>(i)
              + static_cast<std::size_t>(j) * Nx
              + static_cast<std::size_t>(k) * Nx * Ny;

            chars.x_left  = &eigs.x_left[idx];
            chars.x_right = &eigs.x_right[idx];
            chars.x_eigs  = &eigs.x_eigs[idx];

            if constexpr (AETHER_DIM > 1) {
                chars.y_left  = &eigs.y_left[idx];
                chars.y_right = &eigs.y_right[idx];
                chars.y_eigs  = &eigs.y_eigs[idx];
            } else {
                chars.y_left  = nullptr;
                chars.y_right = nullptr;
                chars.y_eigs  = nullptr;
            }

            if constexpr (AETHER_DIM > 2) {
                chars.z_left  = &eigs.z_left[idx];
                chars.z_right = &eigs.z_right[idx];
                chars.z_eigs  = &eigs.z_eigs[idx];
            } else {
                chars.z_left  = nullptr;
                chars.z_right = nullptr;
                chars.z_eigs  = nullptr;
            }

            fill_eigenvectors(prim, chars, gamma);
        }
    );

    Kokkos::fence();

    if (eigs.populated != nullptr) {
        *eigs.populated = true;
    }
}

} // namespace aether::physics::euler