#pragma once

#include "aether/core/prim_layout.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/config.hpp>
#include <aether/core/Kokkos_Policy.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>
#include <aether/core/strides.hpp>  
#include <aether/physics/counts.hpp>

namespace aether::core {

// ============================================================
// Lightweight cell-centered view bundle
// ============================================================
//
// Canonical cell layout from Policy:
//   U(var, k, j, i)
//
// This is mostly a semantic wrapper around the underlying Kokkos view
// plus the associated cell grid metadata.

template<class Policy>
struct CellsViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template CellView<double>;
    using value_type  = typename policy_type::scalar_type;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    CellsViewT() = default;

    AETHER_INLINE
    double& var(int c, int i, int j = 0, int k = 0) const noexcept {
        return data(c, k, j, i);
    }
};


// Const version for read-only kernels.
template<class Policy>
struct CellsConstViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template CellView<const double>;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    CellsConstViewT() = default;

    AETHER_INLINE
    double var(int c, int i, int j = 0, int k = 0) const noexcept {
        return data(c, k, j, i);
    }
};


// ============================================================
// Owned cell-centered field bundle
// ============================================================

template<class Policy, int NCOMP>
struct CellsT {
    using policy_type      = Policy;
    using view_type        = typename policy_type::template CellView<double>;
    using const_view_type  = typename policy_type::template CellView<const double>;
    using mirror_type      = typename policy_type::template CellHostMirror<double>;

    static constexpr int ncomp = NCOMP;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    CellsT() = default;

    explicit CellsT(int nx, int ng)
        requires (Policy::DIMENSION == 1)
        : grid(nx, ng),
          data("cells", NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    explicit CellsT(int nx, int ny, int ng)
        requires (Policy::DIMENSION == 2)
        : grid(nx, ny, ng),
          data("cells", NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    explicit CellsT(int nx, int ny, int nz, int ng)
        requires (Policy::DIMENSION == 3)
        : grid(nx, ny, nz, ng),
          data("cells", NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    [[nodiscard]] AETHER_INLINE
    CellsViewT<Policy> view() const noexcept {
        CellsViewT<Policy> v;
        v.data = data;
        v.grid = grid;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    CellsConstViewT<Policy> const_view() const noexcept {
        CellsConstViewT<Policy> v;
        v.data = data;
        v.grid = grid;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    mirror_type host_mirror() const {
        return Kokkos::create_mirror_view(data);
    }
};


// ============================================================
// Directional cell-centered view bundle
// ============================================================
//
// Canonical directional layout:
//   A(dir, var, k, j, i)
//

template<class Policy>
struct DirCellsViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template DirCellView<double>;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    DirCellsViewT() = default;

    AETHER_INLINE
    double& var(int dir, int c, int i, int j = 0, int k = 0) const noexcept {
        return data(dir, c, k, j, i);
    }
};

template<class Policy>
struct DirCellsConstViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template DirCellView<const double>;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    DirCellsConstViewT() = default;

    AETHER_INLINE
    double var(int dir, int c, int i, int j = 0, int k = 0) const noexcept {
        return data(dir, c, k, j, i);
    }
};

template<class Policy, int NCOMP>
struct DirCellsT {
    using policy_type      = Policy;
    using view_type        = typename policy_type::template DirCellView<double>;
    using mirror_type      = typename policy_type::template DirCellHostMirror<double>;

    static constexpr int ndir = Policy::DIMENSION;
    static constexpr int ncomp = NCOMP;

    view_type data{};
    CellGrid<Policy::DIMENSION> grid{};

    DirCellsT() = default;

    explicit DirCellsT(int nx, int ng)
        requires (Policy::DIMENSION == 1)
        : grid(nx, ng),
          data("dir_cells", ndir, NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    explicit DirCellsT(int nx, int ny, int ng)
        requires (Policy::DIMENSION == 2)
        : grid(nx, ny, ng),
          data("dir_cells", ndir, NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    explicit DirCellsT(int nx, int ny, int nz, int ng)
        requires (Policy::DIMENSION == 3)
        : grid(nx, ny, nz, ng),
          data("dir_cells", ndir, NCOMP, grid.Nz, grid.Ny, grid.Nx) {}

    [[nodiscard]] AETHER_INLINE
    DirCellsViewT<Policy> view() const noexcept {
        DirCellsViewT<Policy> v;
        v.data = data;
        v.grid = grid;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    DirCellsConstViewT<Policy> const_view() const noexcept {
        DirCellsConstViewT<Policy> v;
        v.data = data;
        v.grid = grid;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    mirror_type host_mirror() const {
        return Kokkos::create_mirror_view(data);
    }
};


// ============================================================
// Face-centered view bundles
// ============================================================
//
// Recommended canonical face layout for LayoutRight:
//   F(var, q, k, j, i)
//
// This keeps i as the rightmost / contiguous index.
//

template<class Policy>
struct FaceViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template FaceView<double>;

    view_type data{};
    int Q = 1;

    FaceViewT() = default;

    AETHER_INLINE
    double& var(int c, int q, int i, int j = 0, int k = 0) const noexcept {
        return data(c, q, k, j, i);
    }
};

template<class Policy>
struct FaceConstViewT {
    using policy_type = Policy;
    using view_type   = typename policy_type::template FaceView<const double>;

    view_type data{};
    int Q = 1;

    FaceConstViewT() = default;

    AETHER_INLINE
    double var(int c, int q, int i, int j = 0, int k = 0) const noexcept {
        return data(c, q, k, j, i);
    }
};


// Separate owners for x/y/z face families.
template<class Policy, int NCOMP>
struct FaceArrayX {
    using policy_type = Policy;
    using view_type   = typename policy_type::template FaceView<double>;
    using mirror_type = typename policy_type::template FaceHostMirror<double>;

    view_type data{};
    FaceGridX grid{};
    int Q = 1;

    FaceArrayX() = default;

    explicit FaceArrayX(const CellGrid<Policy::DIMENSION>& cells, int quad)
        : grid(cells), Q(quad),
          data("x_faces", NCOMP, Q, grid.Nz, grid.Ny, grid.Nfx) {}

    [[nodiscard]] AETHER_INLINE
    FaceViewT<Policy> view() const noexcept {
        FaceViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    FaceConstViewT<Policy> const_view() const noexcept {
        FaceConstViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    mirror_type host_mirror() const {
        return Kokkos::create_mirror_view(data);
    }
};

template<class Policy, int NCOMP>
struct FaceArrayY {
    using policy_type = Policy;
    using view_type   = typename policy_type::template FaceView<double>;
    using mirror_type = typename policy_type::template FaceHostMirror<double>;

    view_type data{};
    FaceGridY grid{};
    int Q = 1;

    FaceArrayY() = default;

    explicit FaceArrayY(const CellGrid<Policy::DIMENSION>& cells, int quad)
        : grid(cells), Q(quad),
          data("y_faces", NCOMP, Q, grid.Nz, grid.Nfy, grid.Nx) {}

    [[nodiscard]] AETHER_INLINE
    FaceViewT<Policy> view() const noexcept {
        FaceViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    FaceConstViewT<Policy> const_view() const noexcept {
        FaceConstViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    mirror_type host_mirror() const {
        return Kokkos::create_mirror_view(data);
    }
};

template<class Policy, int NCOMP>
struct FaceArrayZ {
    using policy_type = Policy;
    using view_type   = typename policy_type::template FaceView<double>;
    using mirror_type = typename policy_type::template FaceHostMirror<double>;

    view_type data{};
    FaceGridZ grid{};
    int Q = 1;

    FaceArrayZ() = default;

    explicit FaceArrayZ(const CellGrid<Policy::DIMENSION>& cells, int quad)
        : grid(cells), Q(quad),
          data("z_faces", NCOMP, Q, grid.Nfz, grid.Ny, grid.Nx) {}

    [[nodiscard]] AETHER_INLINE
    FaceViewT<Policy> view() const noexcept {
        FaceViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    FaceConstViewT<Policy> const_view() const noexcept {
        FaceConstViewT<Policy> v;
        v.data = data;
        v.Q = Q;
        return v;
    }

    [[nodiscard]] AETHER_INLINE
    mirror_type host_mirror() const {
        return Kokkos::create_mirror_view(data);
    }
};


// ============================================================
// Basic linear algebra helpers for cell fields
// ============================================================

template<class Policy, int NCOMP>
AETHER_INLINE
void deep_copy(CellsT<Policy, NCOMP>& dst,
               const CellsT<Policy, NCOMP>& src)
{
    Kokkos::deep_copy(dst.data, src.data);
}

template<class Policy, int NCOMP>
void scale(CellsT<Policy, NCOMP>& x, double a)
{
    auto X = x.data;
    Kokkos::parallel_for(
        "scale_cells",
        Kokkos::MDRangePolicy<typename Policy::execution_space, Kokkos::Rank<4>>(
            {0, 0, 0, 0},
            {NCOMP, x.grid.Nz, x.grid.Ny, x.grid.Nx}
        ),
        KOKKOS_LAMBDA(const int c, const int k, const int j, const int i) {
            X(c, k, j, i) *= a;
        }
    );
}

template<class SimT, class ViewX>
void axpy(SimT sim, double a, ViewX x)
{
    auto domain = sim.view();
    auto prim = domain.prim;
    auto cons = domain.cons;
    int numvar = SimT::numvar;

    Kokkos::parallel_for(
        "axpy_cells",
        aether::loops::cells_full(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int c = 0; c < numvar; c++){
                cons(c, k, j, i) += a * x(c, k, j, i);
            }
        }
    );
}


// ============================================================
// Active-build aliases
// ============================================================

using KokkosConfig = aether::kokkos_cfg::KokkosConfig;

using CellsView      = CellsViewT<KokkosConfig>;
using CellsConstView = CellsConstViewT<KokkosConfig>;
using CellsSoA       = CellsT<KokkosConfig, aether::phys_ct::numvar>;

using CharView       = DirCellsViewT<KokkosConfig>;
using CharConstView  = DirCellsConstViewT<KokkosConfig>;
using CharSoA        = DirCellsT<KokkosConfig, aether::phys_ct::numvar>;

using FaceArrayView      = FaceViewT<KokkosConfig>;
using FaceArrayConstView = FaceConstViewT<KokkosConfig>;
using FaceArrayXSoA      = FaceArrayX<KokkosConfig, aether::phys_ct::numvar>;
using FaceArrayYSoA      = FaceArrayY<KokkosConfig, aether::phys_ct::numvar>;
using FaceArrayZSoA      = FaceArrayZ<KokkosConfig, aether::phys_ct::numvar>;

} // namespace aether::core
