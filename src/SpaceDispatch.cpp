#include <Kokkos_Core.hpp>

#include <aether/core/SpaceDispatch.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/simulation.hpp>
#include <aether/physics/counts.hpp>

#include <stdexcept>

namespace loop = aether::loops;
namespace aether::core {

template<sweep_dir dir, class Sim>
AETHER_INLINE void FOG_sweep(Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;
    constexpr int i0 = (dir == sweep_dir::x) ? 1 : 0;
    constexpr int j0 = (dir == sweep_dir::y) ? 1 : 0;
    constexpr int k0 = (dir == sweep_dir::z) ? 1 : 0;

    auto view = sim.view();
    auto prim = view.prim;
    const int quad = view.quad;

    auto FL = [&]() {
        if constexpr (dir == sweep_dir::x) return view.fxL;
        else if constexpr (dir == sweep_dir::y) return view.fyL;
        else return view.fzL;
    }();

    auto FR = [&]() {
        if constexpr (dir == sweep_dir::x) return view.fxR;
        else if constexpr (dir == sweep_dir::y) return view.fyR;
        else return view.fzR;
    }();

    Kokkos::parallel_for(
        "FOG_sweep",
        loop::cells_full(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int c = 0; c < numvar; ++c) {
                const double u = prim(c, k, j, i);
                for (int q = 0; q < quad; ++q) {
                    FR(c, q, k,     j,     i    ) = u;
                    FL(c, q, k + k0, j + j0, i + i0) = u;
                }
            }
        }
    );
}

// ---------- Space solver dispatcher ----------

void Space_solve(Simulation& Sim) {
    switch (Sim.cfg.solve) {
        case solver::fog:
            FOG_sweep<sweep_dir::x>(Sim);
            if constexpr (AETHER_DIM > 1) {
                FOG_sweep<sweep_dir::y>(Sim);
            }
            if constexpr (AETHER_DIM > 2) {
                FOG_sweep<sweep_dir::z>(Sim);
            }
            break;

        default:
            throw std::runtime_error("Space_solve: unknown space solver");
    }
}

} // namespace aether::core