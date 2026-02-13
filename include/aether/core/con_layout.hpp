#pragma once
#include "aether/physics/counts.hpp"
#include <aether/core/config_build.hpp>

namespace aether::con {
constexpr int numvar = aether::phys_ct::numvar;
// Physics kind codes must match config_build.hpp: 1=Euler, 2=SRHD, 3=MHD
template<int DIM, int PHYS> struct Layout;

// ---------- Euler (HD) ----------
template<int DIM>
struct Layout<DIM, 1> {
  static_assert(DIM==1 || DIM==2 || DIM==3, "DIM must be 1,2,3");
  static constexpr int RHO = 0;
  static constexpr int MX  = 1;

  // Presence flags (compile-time)
  static constexpr bool HAS_MY = (numvar >= 4);
  static constexpr bool HAS_MZ = (numvar == 5);

  // Indices (−1 when absent; guarded by HAS_* in if constexpr)
  static constexpr int MY = HAS_MY ? 2 : -1;
  static constexpr int MZ = HAS_MZ ? (HAS_MY ? 3 : 2) : -1;

  static constexpr int E  = HAS_MZ ? 4 : (HAS_MY ? 3 : 2);
  static constexpr int COUNT = E + 1;
};

// ---------- SRHD ----------
template<int DIM>
struct Layout<DIM, 2> {
  static_assert(DIM==1 || DIM==2 || DIM==3, "DIM must be 1,2,3");
  // Always five primitives regardless of DIM
  static constexpr int RHO = 0;
  static constexpr int MX  = 1;
  static constexpr int MY  = 2;
  static constexpr int MZ  = 3;
  static constexpr int E   = 4;
  static constexpr int COUNT = 5;

  static constexpr bool HAS_MY = true;
  static constexpr bool HAS_MZ = true;
};

// -------------------- MHD (placeholder) --------------------



// --------- Convenience alias ---------
using Cons = Layout<AETHER_DIM, AETHER_PHYSICS_KIND>;

} // namespace aether::prim