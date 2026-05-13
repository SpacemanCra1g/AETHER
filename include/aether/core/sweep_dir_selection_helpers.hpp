#pragma once
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/prim_layout.hpp>

using P = aether::prim::Prim;
namespace aether::core{


// ============================================================
// Velocity remapping by sweep direction
// hll() expects normal velocity in vx, transverse in vy/vz
// ============================================================
template<sweep_dir dir>
struct VelMap;

template<>
struct VelMap<sweep_dir::x> {
    static constexpr int VN  = P::VX;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VZ;
};

template<>
struct VelMap<sweep_dir::y> {
    static constexpr int VN  = P::VY;
    static constexpr int VT1 = P::VX;
    static constexpr int VT2 = P::VZ;
};

template<>
struct VelMap<sweep_dir::z> {
    static constexpr int VN  = P::VZ;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VX;
};

// ============================================================
// Select the face-state buffers for a sweep
// ============================================================

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_left_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fxL;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fyL;
    }
    else {
        return v.fzL;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_right_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fxR;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fyR;
    }
    else {
        return v.fzR;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto flux_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.fx;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.fy;
    }
    else {
        return v.fz;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto source_flux_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.source_flux_x;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.source_flux_y;
    }
    else {
        return v.source_flux_z;
    }
}

template<sweep_dir dir, class View>
AETHER_INLINE auto source_view(View& v) {
    if constexpr (dir == sweep_dir::x) {
        return v.sources_x;
    }
    else if constexpr (dir == sweep_dir::y) {
        return v.sources_y;
    }
    else {
        return v.sources_z;
    }
}
}
