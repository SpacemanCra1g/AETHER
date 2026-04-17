#pragma once

#include <Kokkos_Core.hpp>
#include <aether/physics/counts.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/config_build.hpp>

namespace aether::kokkos_cfg {

// Backend / type policy for AETHER data storage.
// DIM  = problem dimension (1,2,3)
// PHYS = physics kind code from config_build.hpp
// ExecSpace = Kokkos execution space
// ArrayLayout = Kokkos layout policy
// Scalar = floating-point storage type
// Index  = integer index type

template<int DIM, int PHYS
    , class ExecSpace = Kokkos::DefaultExecutionSpace
    , class ArrayLayout = Kokkos::LayoutRight
    , class Scalar      = double
    , class Index       = int >
struct Policy {

    // ---------- Compile-time identifiers ----------
    static constexpr int DIMENSION    = DIM;
    static constexpr int PHYSICS_KIND = PHYS;

    // Physics-dependent variable count.
    //
    // For now this assumes the same count for the major field bundles.
    // Later, if you want separate primitive / conservative / auxiliary counts,
    // we can split these cleanly.
    static constexpr int NUMVAR = aether::phys_ct::numvar;

    // Primitive-variable count, if you want the policy to expose it directly.
    static constexpr int NPRIM = aether::prim::Layout<DIM, PHYS>::COUNT;
    static constexpr int NCONS  = NPRIM;

    // ---------- Kokkos backend types ----------
    using execution_space = ExecSpace;
    using memory_space    = typename execution_space::memory_space;
    using device_type     = Kokkos::Device<execution_space, memory_space>;
    using array_layout    = ArrayLayout;


    using scalar_type = Scalar;
    using index_type  = Index;
    using size_type   = std::size_t;

    // ---------- Canonical view ranks ----------
    //
    // Unified rank model:
    //   cell-centered      : (var, k, j, i)
    //   face-centered      : (var, q, k, j, i)
    //   directional cells  : (dir, var, k, j, i)
    //
    // Lower dimensions should use extent-1 for unused axes.
    static constexpr int CELL_RANK      = 4;
    static constexpr int FACE_RANK      = 5;
    static constexpr int DIR_CELL_RANK  = 5;

    // ---------- Canonical Kokkos view aliases ----------
    //
    // Cell-centered field bundle:
    //   U(v,i,j,k), W(v,i,j,k), source(v,i,j,k), etc.
    template<class T = scalar_type>
    using CellView = Kokkos::View<T****, array_layout, device_type>;

    // Face-centered field bundle:
    //   Fx(v,i,j,k,q), Fy(v,i,j,k,q), Fz(v,i,j,k,q)
    //
    // The meaning of (i,j,k) depends on the face family:
    //   x-face: i spans x-faces, j/k are cell-indexed transverse directions
    //   y-face: j spans y-faces
    //   z-face: k spans z-faces
    template<class T = scalar_type>
    using FaceView = Kokkos::View<T*****, array_layout, device_type>;

    // Directional cell-centered bundle:
    //   A(dir,v,i,j,k), eigen(dir,v,i,j,k), characteristic(dir,v,i,j,k), etc.
    template<class T = scalar_type>
    using DirCellView = Kokkos::View<T*****, array_layout, device_type>;

    // Integer versions are often useful for masks / flags / indexing helpers.
    using IntCellView     = CellView<index_type>;
    using IntFaceView     = FaceView<index_type>;
    using IntDirCellView  = DirCellView<index_type>;

    // ---------- Unmanaged view aliases ----------
    //
    // These can be useful later if you want lightweight wrappers or to pass
    // non-owning views around without reallocating / refcounting.
    template<class T = scalar_type>
    using CellViewUnmanaged =
        Kokkos::View<T****, array_layout, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template<class T = scalar_type>
    using FaceViewUnmanaged =
        Kokkos::View<T*****, array_layout, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template<class T = scalar_type>
    using DirCellViewUnmanaged =
        Kokkos::View<T*****, array_layout, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // ---------- Mirror aliases ----------
    //
    // These aliases expose the host mirror type corresponding to each managed
    // view. Very handy for initialization, debugging, and snapshot I/O.
template<class T = scalar_type>
using CellHostMirror = typename CellView<T>::host_mirror_type;

template<class T = scalar_type>
using FaceHostMirror = typename FaceView<T>::host_mirror_type;

template<class T = scalar_type>
using DirCellHostMirror = typename DirCellView<T>::host_mirror_type;
    // ---------- Small helper constexprs ----------
    static constexpr bool HAS_Y = (DIM >= 2);
    static constexpr bool HAS_Z = (DIM >= 3);
};

// Convenience alias for the active build configuration.
using KokkosConfig = Policy<AETHER_DIM, AETHER_PHYSICS_KIND>;

} // namespace aether::kokkos_cfg