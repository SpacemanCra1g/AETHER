#pragma once
#include <array>
#include <aether/core/enums.hpp>
#include <aether/core/config.hpp>

namespace aether::core {

template<int NVAR>
struct CellAccessor {
    // pointer storing raw data
    double* AETHER_RESTRICT* comp = nullptr; 

    // Flat index of the center cell
    std::size_t base = 0;

    // Strides in flat index space 
    std::ptrdiff_t sx = 1;
    std::ptrdiff_t sy = 0;
    std::ptrdiff_t sz = 0;

    CellAccessor() = default;

    // Alright, let's see if this constructor finally works
    AETHER_INLINE CellAccessor(double* AETHER_RESTRICT* comp_ptrs,
                            std::size_t base_idx,
                            std::ptrdiff_t sx_,
                            std::ptrdiff_t sy_,
                            std::ptrdiff_t sz_) noexcept
    : comp(comp_ptrs), base(base_idx), sx(sx_), sy(sy_), sz(sz_) {}

    // Construct from std::array of pointers (const)
    AETHER_INLINE CellAccessor(const std::array<const double*, NVAR>& comp_arr,
                              std::size_t base_idx,
                              std::ptrdiff_t sx_,
                              std::ptrdiff_t sy_,
                              std::ptrdiff_t sz_) noexcept
        : comp(comp_arr.data()), base(base_idx), sx(sx_), sy(sy_), sz(sz_) {}

    // Construct from std::array of pointers (non-const)
    AETHER_INLINE CellAccessor(const std::array<double*, NVAR>& comp_arr,
                              std::size_t base_idx,
                              std::ptrdiff_t sx_,
                              std::ptrdiff_t sy_,
                              std::ptrdiff_t sz_) noexcept
        : comp(reinterpret_cast<double* AETHER_RESTRICT*>(comp_arr.data())),
          base(base_idx), sx(sx_), sy(sy_), sz(sz_) {}

    // Offsets are in cell units; stride converts to flat index offset.
    AETHER_INLINE double get(const int var,
                             const std::ptrdiff_t ox,
                             const std::ptrdiff_t oy,
                             const std::ptrdiff_t oz) const noexcept
    {
        const std::ptrdiff_t idx =
            static_cast<std::ptrdiff_t>(base) + ox*sx + oy*sy + oz*sz;
        return comp[var][static_cast<std::size_t>(idx)];
    }

    // Scary reference form, don't want to enable unless I have to
/*    AETHER_INLINE double& ref(const int var,
                              const std::ptrdiff_t ox,
                              const std::ptrdiff_t oy,
                              const std::ptrdiff_t oz) const noexcept
    {
        const std::ptrdiff_t idx =
            static_cast<std::ptrdiff_t>(base) + ox*sx + oy*sy + oz*sz;
        return const_cast<double*>(comp[var])[static_cast<std::size_t>(idx)];
    }
*/
};


//---------- Stencil1D ----------
template<int R, int NVAR, sweep_dir dir>
struct Stencil1D {
    CellAccessor<NVAR> A{};
    std::ptrdiff_t s = 1; // stride along chosen direction

    static constexpr int radius = R;
    static constexpr int width  = 2*R + 1;

    Stencil1D() = default;

    AETHER_INLINE explicit Stencil1D(const CellAccessor<NVAR>& acc) noexcept
        : A(acc)
    {
        if constexpr (dir == sweep_dir::x)      s = A.sx;
        else if constexpr (dir == sweep_dir::y) s = A.sy;
        else                                    s = A.sz;
    }

    // Load variable at 1D offset along dir
    AETHER_INLINE double get(const int var, const int off) const noexcept {
        const std::ptrdiff_t idx =
            static_cast<std::ptrdiff_t>(A.base) + static_cast<std::ptrdiff_t>(off)*s;
        return A.comp[var][static_cast<std::size_t>(idx)];
    }

    
/*  // Writable reference version see above
    AETHER_INLINE double& ref(const int var, const int off) const noexcept {
        const std::ptrdiff_t idx =
            static_cast<std::ptrdiff_t>(A.base) + static_cast<std::ptrdiff_t>(off)*s;
        return const_cast<double*>(A.comp[var])[static_cast<std::size_t>(idx)];
    }
*/
};

} // namespace aether::core
