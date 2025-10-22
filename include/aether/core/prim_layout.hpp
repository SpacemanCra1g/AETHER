#include <aether/core/config_build.hpp>

namespace aether::prim {

// Physics kind codes must match config_build.hpp: 1=Euler, 2=SRHD, 3=MHD
template<int DIM, int PHYS> struct Layout;

// ---------- Euler (HD) ----------
template<int DIM>
struct Layout<DIM, 1> {
  static_assert(DIM==1 || DIM==2 || DIM==3, "DIM must be 1,2,3");
  static constexpr int RHO = 0;
  static constexpr int VX  = 1;

  // Presence flags (compile-time)
  static constexpr bool HAS_VY = (DIM >= 2);
  static constexpr bool HAS_VZ = (DIM == 3);

  // Indices (−1 when absent; guarded by HAS_* in if constexpr)
  static constexpr int VY = HAS_VY ? 2 : -1;
  static constexpr int VZ = HAS_VZ ? (HAS_VY ? 3 : 2) : -1;

  static constexpr int P  = HAS_VZ ? 4 : (HAS_VY ? 3 : 2);
  static constexpr int COUNT = P + 1;
};

// ---------- SRHD ----------
template<int DIM>
struct Layout<DIM, 2> {
  static_assert(DIM==1 || DIM==2 || DIM==3, "DIM must be 1,2,3");
  // Always five primitives regardless of DIM
  static constexpr int RHO = 0;
  static constexpr int VX  = 1;
  static constexpr int VY  = 2;
  static constexpr int VZ  = 3;
  static constexpr int P   = 4;
  static constexpr int COUNT = 5;

  static constexpr bool HAS_VY = true;
  static constexpr bool HAS_VZ = true;
};

// -------------------- MHD (placeholder) --------------------



// --------- Convenience alias ---------
using Prim = Layout<AETHER_DIM, AETHER_PHYSICS_KIND>;

} // namespace aether::prim