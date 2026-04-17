#pragma once

#include <Kokkos_Core.hpp>

#include <aether/core/enums.hpp>
#include <aether/core/config.hpp>
#include <aether/core/Kokkos_Policy.hpp>

namespace aether::core {

// ============================================================
// Cell-centered accessor
// ============================================================
//
// Canonical cell layout from Policy:
//   U(var, k, j, i)
//
// This accessor stores a view handle plus a center cell (i0,j0,k0)
// and provides relative-offset access around that cell.
//

template<class Policy>
struct CellAccessor {
    using policy_type = Policy;
    using view_type   = typename policy_type::template CellView<const double>;
    using value_type  = typename policy_type::scalar_type;

    view_type U{};
    int i0 = 0;
    int j0 = 0;
    int k0 = 0;

    CellAccessor() = default;

    AETHER_INLINE
    CellAccessor(const view_type& U_, int i, int j = 0, int k = 0) noexcept
        : U(U_), i0(i), j0(j), k0(k) {}

    AETHER_INLINE
    value_type get(const int var,
                   const int ox = 0,
                   const int oy = 0,
                   const int oz = 0) const noexcept
    {
        return U(var, k0 + oz, j0 + oy, i0 + ox);
    }
};


// Writable version if/when you decide you want it.
template<class Policy>
struct CellAccessorMut {
    using policy_type = Policy;
    using view_type   = typename policy_type::template CellView<double>;
    using value_type  = typename policy_type::scalar_type;

    view_type U{};
    int i0 = 0;
    int j0 = 0;
    int k0 = 0;

    CellAccessorMut() = default;

    AETHER_INLINE
    CellAccessorMut(const view_type& U_, int i, int j = 0, int k = 0) noexcept
        : U(U_), i0(i), j0(j), k0(k) {}

    AETHER_INLINE
    value_type get(const int var,
                   const int ox = 0,
                   const int oy = 0,
                   const int oz = 0) const noexcept
    {
        return U(var, k0 + oz, j0 + oy, i0 + ox);
    }

    AETHER_INLINE
    value_type& ref(const int var,
                    const int ox = 0,
                    const int oy = 0,
                    const int oz = 0) const noexcept
    {
        return U(var, k0 + oz, j0 + oy, i0 + ox);
    }
};


// ============================================================
// 1D stencil accessor along a chosen sweep direction
// ============================================================
//
// dir selects which geometric axis the 1D offsets follow.
// R is the stencil radius.
//
// Example:
//   Stencil1D<2, Policy, sweep_dir::x> sx(acc);
//   sx.get(var, -2), sx.get(var, -1), ..., sx.get(var, +2)
//

template<int R, class Policy, sweep_dir dir>
struct Stencil1D {
    using accessor_type = CellAccessor<Policy>;
    using value_type    = typename Policy::scalar_type;

    accessor_type A{};

    static constexpr int radius = R;
    static constexpr int width  = 2*R + 1;

    Stencil1D() = default;

    AETHER_INLINE
    explicit Stencil1D(const accessor_type& acc) noexcept : A(acc) {}

    AETHER_INLINE
    value_type get(const int var, const int off) const noexcept {
        if constexpr (dir == sweep_dir::x) {
            return A.get(var, off, 0, 0);
        } else if constexpr (dir == sweep_dir::y) {
            return A.get(var, 0, off, 0);
        } else {
            return A.get(var, 0, 0, off);
        }
    }
};


template<int R, class Policy, sweep_dir dir>
struct Stencil1DMut {
    using accessor_type = CellAccessorMut<Policy>;
    using value_type    = typename Policy::scalar_type;

    accessor_type A{};

    static constexpr int radius = R;
    static constexpr int width  = 2*R + 1;

    Stencil1DMut() = default;

    AETHER_INLINE
    explicit Stencil1DMut(const accessor_type& acc) noexcept : A(acc) {}

    AETHER_INLINE
    value_type get(const int var, const int off) const noexcept {
        if constexpr (dir == sweep_dir::x) {
            return A.get(var, off, 0, 0);
        } else if constexpr (dir == sweep_dir::y) {
            return A.get(var, 0, off, 0);
        } else {
            return A.get(var, 0, 0, off);
        }
    }

    AETHER_INLINE
    value_type& ref(const int var, const int off) const noexcept {
        if constexpr (dir == sweep_dir::x) {
            return A.ref(var, off, 0, 0);
        } else if constexpr (dir == sweep_dir::y) {
            return A.ref(var, 0, off, 0);
        } else {
            return A.ref(var, 0, 0, off);
        }
    }
};


// ============================================================
// Cell domain descriptor
// ============================================================
//
// This replaces the "stride" role of the old extents/access coupling.
// It stores padded and interior sizes only; no flattening logic.
//

template<int DIM>
struct CellGrid;

template<>
struct CellGrid<1> {
    int nx = 0;
    int ny = 1;
    int nz = 1;

    int ng = 0;

    int Nx = 0;
    int Ny = 1;
    int Nz = 1;

    CellGrid() = default;

    explicit CellGrid(int nx_, int ng_) noexcept
        : nx(nx_), ng(ng_), Nx(nx_ + 2*ng_) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx; }

    AETHER_INLINE int jbegin() const noexcept { return 0; }
    AETHER_INLINE int jend()   const noexcept { return 1; }

    AETHER_INLINE int kbegin() const noexcept { return 0; }
    AETHER_INLINE int kend()   const noexcept { return 1; }
};

template<>
struct CellGrid<2> {
    int nx = 0, ny = 0;
    int nz = 1;

    int ng = 0;

    int Nx = 0, Ny = 0;
    int Nz = 1;

    CellGrid() = default;

    explicit CellGrid(int nx_, int ny_, int ng_) noexcept
        : nx(nx_), ny(ny_), ng(ng_),
          Nx(nx_ + 2*ng_), Ny(ny_ + 2*ng_) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx; }

    AETHER_INLINE int jbegin() const noexcept { return ng; }
    AETHER_INLINE int jend()   const noexcept { return ng + ny; }

    AETHER_INLINE int kbegin() const noexcept { return 0; }
    AETHER_INLINE int kend()   const noexcept { return 1; }
};

template<>
struct CellGrid<3> {
    int nx = 0, ny = 0, nz = 0;
    int ng = 0;
    int Nx = 0, Ny = 0, Nz = 0;

    CellGrid() = default;

    explicit CellGrid(int nx_, int ny_, int nz_, int ng_) noexcept
        : nx(nx_), ny(ny_), nz(nz_), ng(ng_),
          Nx(nx_ + 2*ng_), Ny(ny_ + 2*ng_), Nz(nz_ + 2*ng_) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx; }

    AETHER_INLINE int jbegin() const noexcept { return ng; }
    AETHER_INLINE int jend()   const noexcept { return ng + ny; }

    AETHER_INLINE int kbegin() const noexcept { return ng; }
    AETHER_INLINE int kend()   const noexcept { return ng + nz; }
};

using Cells = CellGrid<AETHER_DIM>;


// ============================================================
// Face-grid descriptors
// ============================================================
//
// Canonical face view shape (recommended):
//   F(var, q, k, j, i)
//
// where the face family determines which spatial axis is staggered.
//
// These structs describe extents and valid ranges only.
// They do NOT flatten indices.
//

struct FaceGridX {
    int nx = 0, ny = 1, nz = 1;
    int ng = 0;

    int Nx = 0, Ny = 1, Nz = 1;   // padded cell extents
    int Nfx = 0;                  // padded x-face extent in i

    FaceGridX() = default;

    explicit FaceGridX(const CellGrid<1>& c) noexcept
        : nx(c.nx), ny(1), nz(1), ng(c.ng),
          Nx(c.Nx), Ny(1), Nz(1),
          Nfx(c.Nx + 1) {}

    explicit FaceGridX(const CellGrid<2>& c) noexcept
        : nx(c.nx), ny(c.ny), nz(1), ng(c.ng),
          Nx(c.Nx), Ny(c.Ny), Nz(1),
          Nfx(c.Nx + 1) {}

    explicit FaceGridX(const CellGrid<3>& c) noexcept
        : nx(c.nx), ny(c.ny), nz(c.nz), ng(c.ng),
          Nx(c.Nx), Ny(c.Ny), Nz(c.Nz),
          Nfx(c.Nx + 1) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx + 1; }

    AETHER_INLINE int jbegin() const noexcept { return (Ny > 1 ? ng : 0); }
    AETHER_INLINE int jend()   const noexcept { return (Ny > 1 ? ng + ny : 1); }

    AETHER_INLINE int kbegin() const noexcept { return (Nz > 1 ? ng : 0); }
    AETHER_INLINE int kend()   const noexcept { return (Nz > 1 ? ng + nz : 1); }
};

struct FaceGridY {
    int nx = 0, ny = 0, nz = 1;
    int ng = 0;

    int Nx = 0, Ny = 0, Nz = 1;
    int Nfy = 0;

    FaceGridY() = default;

    explicit FaceGridY(const CellGrid<2>& c) noexcept
        : nx(c.nx), ny(c.ny), nz(1), ng(c.ng),
          Nx(c.Nx), Ny(c.Ny), Nz(1),
          Nfy(c.Ny + 1) {}

    explicit FaceGridY(const CellGrid<3>& c) noexcept
        : nx(c.nx), ny(c.ny), nz(c.nz), ng(c.ng),
          Nx(c.Nx), Ny(c.Ny), Nz(c.Nz),
          Nfy(c.Ny + 1) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx; }

    AETHER_INLINE int jbegin() const noexcept { return ng; }
    AETHER_INLINE int jend()   const noexcept { return ng + ny + 1; }

    AETHER_INLINE int kbegin() const noexcept { return (Nz > 1 ? ng : 0); }
    AETHER_INLINE int kend()   const noexcept { return (Nz > 1 ? ng + nz : 1); }
};

struct FaceGridZ {
    int nx = 0, ny = 0, nz = 0;
    int ng = 0;

    int Nx = 0, Ny = 0, Nz = 0;
    int Nfz = 0;

    FaceGridZ() = default;

    explicit FaceGridZ(const CellGrid<3>& c) noexcept
        : nx(c.nx), ny(c.ny), nz(c.nz), ng(c.ng),
          Nx(c.Nx), Ny(c.Ny), Nz(c.Nz),
          Nfz(c.Nz + 1) {}

    AETHER_INLINE int ibegin() const noexcept { return ng; }
    AETHER_INLINE int iend()   const noexcept { return ng + nx; }

    AETHER_INLINE int jbegin() const noexcept { return ng; }
    AETHER_INLINE int jend()   const noexcept { return ng + ny; }

    AETHER_INLINE int kbegin() const noexcept { return ng; }
    AETHER_INLINE int kend()   const noexcept { return ng + nz + 1; }
};

} // namespace aether::core