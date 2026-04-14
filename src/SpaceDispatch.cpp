#include "aether/physics/euler/variable_structs.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/SpaceDispatch.hpp>
#include <aether/core/Kokkos_loopBounds.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/simulation.hpp>
#include <aether/physics/counts.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/char_struct.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/slope_limiters.hpp>

#include <stdexcept>

namespace loop = aether::loops;
using chars = aether::core::one_cell_spectral_container;
using vec = aether::math::Vec<aether::phys_ct::numvar>;
using prims = aether::phys::prims;
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
        // TODO:: This should be a 1 cell halo ring
        loop::cells_full(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int c = 0; c < numvar; ++c) {
                const double u = prim(c, k, j, i);
                for (int q = 0; q < quad; ++q) {
                    FR(c, q, k     , j     , i     ) = u;
                    FL(c, q, k + k0, j + j0, i + i0) = u;
                }
            }
        }
    );
}

template<limiter TVD, int dim, class Sim>
AETHER_INLINE void PLM_sweep(Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;

    auto view = sim.view();
    auto prim = view.prim;

    double inv_dx = 1.0/view.dx;
    double inv_dy = 1.0/view.dy;
    double dtx = view.dt * inv_dx;
    

    Kokkos::parallel_for(
        "PLM_sweep",
        // TODO:: This should be a 1 cell halo ring
        loops::cells_halo2(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            double dty = view.dt * inv_dy;
            vec p_vec, p_vec_L, p_vec_R;
            prims p;
            chars Eigs;
            for (int c = 0; c < numvar; ++c) {
                p_vec[c] = prim(c,k,j,i);
                p_vec_L[c] = prim(c,k,j,i-1);
                p_vec_R[c] = prim(c,k,j,i+1);
            }
            p.rho = p_vec[0];
            p.vx = p_vec[1];
            if constexpr (P::HAS_VY) p.vy = p_vec[P::VY];
            if constexpr (P::HAS_VZ) p.vz = p_vec[P::VZ];
            p.p = p_vec[P::P];

            fill_eigenvectors(p, Eigs, view.gamma);
            vec d_w;

            if constexpr (TVD == limiter::minmod) {
                d_w =  minmod( Eigs.x.left *(p_vec - p_vec_L), Eigs.x.left *(p_vec_R - p_vec) );
            } else if constexpr (TVD == limiter::mc) {
                d_w = mc( 0.5*Eigs.x.left * (p_vec_R - p_vec_L) , 2.0*Eigs.x.left * (p_vec_R - p_vec), 2.0*Eigs.x.left * (p_vec - p_vec_L)) ;
            } else if constexpr (TVD == limiter::vanleer) {
                d_w =  van_leer(Eigs.x.left *(p_vec - p_vec_L), Eigs.x.left *(p_vec_R - p_vec));
            }

            p_vec_L = p_vec;
            p_vec_R = p_vec;

            for (int c = 0; c < P::COUNT; c++){
                double eig = Eigs.x.lambda(c);
                if (eig >= 0.0){
                    p_vec_R += 0.5*(1.0 - eig*dtx)*d_w[c]*col(Eigs.x.right,c);
                } else {
                    p_vec_L += 0.5*(-1.0 - eig*dtx)*d_w[c]*col(Eigs.x.right,c);
                }
            }

            for (int c = 0; c < numvar; ++c) {
                for (int q = 0; q < view.quad; ++q) {
                    view.fxR(c, q, k, j, i) = p_vec_L[c];
                    view.fxL(c, q, k, j, i+1) = p_vec_R[c];
                }
            }

            // These are the y-sweeps for PLM
            if constexpr (dim > 1) {
                for (int c = 0; c < numvar; ++c) {
                    p_vec_L[c] = prim(c,k,j-1,i);
                    p_vec_R[c] = prim(c,k,j+1,i);
            }

            if constexpr (TVD == limiter::minmod) {
                d_w =  minmod( Eigs.y.left *(p_vec - p_vec_L), Eigs.y.left *(p_vec_R - p_vec) );
            } else if constexpr (TVD == limiter::mc) {
                d_w = mc( 0.5*Eigs.y.left * (p_vec_R - p_vec_L) , 2.0*Eigs.y.left * (p_vec_R - p_vec), 2.0*Eigs.y.left * (p_vec - p_vec_L)) ;
            } else if constexpr (TVD == limiter::vanleer) {
                d_w =  van_leer(Eigs.y.left *(p_vec - p_vec_L), Eigs.y.left *(p_vec_R - p_vec));
            }

            p_vec_L = p_vec;
            p_vec_R = p_vec;

            for (int c = 0; c < P::COUNT; c++){
                double eig = Eigs.y.lambda(c);
                if (eig >= 0.0){
                    p_vec_R += 0.5*(1.0 - eig*dty)*d_w[c]*col(Eigs.y.right,c);
                } else {
                    p_vec_L += 0.5*(-1.0 - eig*dty)*d_w[c]*col(Eigs.y.right,c);
                }
            }

            for (int c = 0; c < numvar; ++c) {
                for (int q = 0; q < view.quad; ++q) {
                    view.fyR(c, q, k, j, i) = p_vec_L[c];
                    view.fyL(c, q, k, j+1, i) = p_vec_R[c];
                }
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
        
        case solver::plm:
            PLM_sweep<limiter::vanleer, AETHER_DIM>(Sim);
            break;
        default:
            throw std::runtime_error("Space_solve: unknown space solver");
    }
}

} // namespace aether::core