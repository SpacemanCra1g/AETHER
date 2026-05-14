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

using chars = aether::core::one_cell_spectral_container;
using vec   = aether::math::Vec<aether::phys_ct::numvar - 1>;
using prims = aether::phys::prims;

namespace aether::core {

using P = aether::prim::Prim;

template<sweep_dir dir, class Sim>
AETHER_INLINE void FOG_sweep(Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;
    constexpr int i0 = (dir == sweep_dir::x) ? 1 : 0;
    constexpr int j0 = (dir == sweep_dir::y) ? 1 : 0;
    constexpr int k0 = (dir == sweep_dir::z) ? 1 : 0;

    auto view = sim.view();
    auto prim = view.prim;
    const int quad = view.quad;

	auto Src = [&]() {
        if constexpr (dir == sweep_dir::x) return view.sources_x;
        else if constexpr (dir == sweep_dir::y) return view.sources_y;
        else return view.sources_z;
    }();

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

	// The FR and FL arrays store the interpolated reconstructions at cell interfaces
    Kokkos::parallel_for(
        "FOG_sweep",
        loops::solver_sweep_policy(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int c = 0; c < numvar-1; ++c) {
                const double u = prim(c, k, j, i);
                for (int q = 0; q < quad; ++q) {
                    FR(c, q, k,      j,      i)      = u;
                    FL(c, q, k + k0, j + j0, i + i0) = u;
                }
            }
			for (int q = 0; q < quad; ++q) {
                    FR(P::EINT, q, k,      j,      i)      = prim(P::EINT, k, j, i) * prim(P::RHO, k, j, i);
                    FL(P::EINT, q, k + k0, j + j0, i + i0) = prim(P::EINT, k, j, i) * prim(P::RHO, k, j, i);
            }
			// Track the interpolated pressure terms for the Eint source component
			Src(0,k,j,i) = prim(P::P, k, j, i);
        }
    );
}

template<limiter TVD, int dim, class Sim>
AETHER_INLINE void PLM_sweep(Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;

    auto view = sim.view();
    auto prim = view.prim;
	auto x_src = view.sources_x;

    const double inv_dx = 1.0 / view.dx;
    const double inv_dy = 1.0 / view.dy;
    const double inv_dz = 1.0 / view.dz;
    const double dtx    = view.dt * inv_dx;

    Kokkos::parallel_for(
        "PLM_sweep",
        loops::solver_sweep_policy(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            [[maybe_unused]] const double dty = view.dt * inv_dy;
            [[maybe_unused]] const double dtz = view.dt * inv_dz;

            vec p_vec, p_vec_L, p_vec_R;
            prims p;
            chars Eigs;

            for (int c = 0; c < numvar-1; ++c) {
                p_vec[c]   = prim(c, k, j, i);
                p_vec_L[c] = prim(c, k, j, i - 1);
                p_vec_R[c] = prim(c, k, j, i + 1);
            }

            p.rho = p_vec[0];
            p.vx  = p_vec[1];
            if constexpr (P::HAS_VY) p.vy = p_vec[P::VY];
            if constexpr (P::HAS_VZ) p.vz = p_vec[P::VZ];
            p.p = p_vec[P::P];

			double eint = prim(P::EINT,k,j,i) * prim(P::RHO,k,j,i);
			double eintL = prim(P::EINT,k,j,i-1) * prim(P::RHO,k,j,i-1);
			double eintR = prim(P::EINT,k,j,i+1) * prim(P::RHO,k,j,i+1);

            fill_eigenvectors(p, Eigs, view.gamma);

            vec d_w;
			double dw_eint;

            if constexpr (TVD == limiter::minmod) {
                d_w = minmod(Eigs.x.left * (p_vec - p_vec_L),
                             Eigs.x.left * (p_vec_R - p_vec));
				dw_eint = minmod(eint-eintL,eintR-eint);
            } else if constexpr (TVD == limiter::mc) {
                d_w = mc(0.5 * Eigs.x.left * (p_vec_R - p_vec_L),
                         2.0 * Eigs.x.left * (p_vec_R - p_vec),
                         2.0 * Eigs.x.left * (p_vec - p_vec_L));
				dw_eint = mc(0.5*(eintR-eintL), 2.0*(eintR - eint), 2.0*(eint - eintL));
            } else if constexpr (TVD == limiter::vanleer) {
                d_w = van_leer(Eigs.x.left * (p_vec - p_vec_L),
                               Eigs.x.left * (p_vec_R - p_vec));
				dw_eint = van_leer( (eint- eintL), (eintR - eint) );
            }

            p_vec_L = p_vec;
            p_vec_R = p_vec;

			// track the internal energy
			double eintL_old = eintL;
			double eintR_old = eintR;

			eintL = eint - 0.5 * dw_eint;
			eintR = eint + 0.5 * dw_eint;

			// Characteristic tracing on the regular variables
            for (int c = 0; c < numvar-1; ++c) {
                const double eig = Eigs.x.lambda(c);
                if (eig >= 0.0) {
                    p_vec_R += 0.5 * (1.0 - eig * dtx) * d_w[c] * col(Eigs.x.right, c);
                } else {
                    p_vec_L += 0.5 * (-1.0 - eig * dtx) * d_w[c] * col(Eigs.x.right, c);
                }
            }

			// characteristic trace on eint
			double u = prim(P::VX,k,j,i);
			eintR -= 0.5*( (u >= 0.0) ? u : 0.0 ) * dtx * dw_eint;
			eintL -= 0.5*( (u <= 0.0) ? u : 0.0 ) * dtx * dw_eint;

			// Ensure interp points lie between cell centered values
			if (eintL < std::fmin(eint, eintL_old) || eintL > std::fmax(eint, eintL_old) ){
				eintL = eint;
			}
			if (eintR < std::fmin(eint, eintR_old) || eintR > std::fmax(eint, eintR_old) ){
				eintR = eint;
			}

            for (int c = 0; c < numvar-1; ++c) {
                for (int q = 0; q < view.quad; ++q) {
                    view.fxR(c, q, k, j, i)   = p_vec_L[c];
                    view.fxL(c, q, k, j, i+1) = p_vec_R[c];
                }
            }
			for (int q = 0; q < view.quad; ++q) view.fxR(P::EINT,q,k,j,i)   = eintL;
			for (int q = 0; q < view.quad; ++q) view.fxL(P::EINT,q,k,j,i+1) = eintR;

			x_src(0,k,j,i) = 0.5 * (p_vec_L[P::P] + p_vec_R[P::P]);

            if constexpr (dim > 1) {
                for (int c = 0; c < numvar-1; ++c) {
                    p_vec_L[c] = prim(c, k, j - 1, i);
                    p_vec_R[c] = prim(c, k, j + 1, i);
                }

                if constexpr (TVD == limiter::minmod) {
                    d_w = minmod(Eigs.y.left * (p_vec - p_vec_L),
                                 Eigs.y.left * (p_vec_R - p_vec));
                } else if constexpr (TVD == limiter::mc) {
                    d_w = mc(0.5 * Eigs.y.left * (p_vec_R - p_vec_L),
                             2.0 * Eigs.y.left * (p_vec_R - p_vec),
                             2.0 * Eigs.y.left * (p_vec - p_vec_L));
                } else if constexpr (TVD == limiter::vanleer) {
                    d_w = van_leer(Eigs.y.left * (p_vec - p_vec_L),
                                   Eigs.y.left * (p_vec_R - p_vec));
                }

                p_vec_L = p_vec;
                p_vec_R = p_vec;

                for (int c = 0; c < numvar-1; ++c) {
                    const double eig = Eigs.y.lambda(c);
                    if (eig >= 0.0) {
                        p_vec_R += 0.5 * (1.0 - eig * dty) * d_w[c] * col(Eigs.y.right, c);
                    } else {
                        p_vec_L += 0.5 * (-1.0 - eig * dty) * d_w[c] * col(Eigs.y.right, c);
                    }
                }

                for (int c = 0; c < numvar-1; ++c) {
                    for (int q = 0; q < view.quad; ++q) {
                        view.fyR(c, q, k, j,   i) = p_vec_L[c];
                        view.fyL(c, q, k, j+1, i) = p_vec_R[c];
                    }
                }
            }

            if constexpr (dim > 2) {
                for (int c = 0; c < numvar-1; ++c) {
                    p_vec_L[c] = prim(c, k - 1, j, i);
                    p_vec_R[c] = prim(c, k + 1, j, i);
                }

                if constexpr (TVD == limiter::minmod) {
                    d_w = minmod(Eigs.z.left * (p_vec - p_vec_L),
                                 Eigs.z.left * (p_vec_R - p_vec));
                } else if constexpr (TVD == limiter::mc) {
                    d_w = mc(0.5 * Eigs.z.left * (p_vec_R - p_vec_L),
                             2.0 * Eigs.z.left * (p_vec_R - p_vec),
                             2.0 * Eigs.z.left * (p_vec - p_vec_L));
                } else if constexpr (TVD == limiter::vanleer) {
                    d_w = van_leer(Eigs.z.left * (p_vec - p_vec_L),
                                   Eigs.z.left * (p_vec_R - p_vec));
                }

                p_vec_L = p_vec;
                p_vec_R = p_vec;

                for (int c = 0; c < numvar-1; ++c) {
                    const double eig = Eigs.z.lambda(c);
                    if (eig >= 0.0) {
                        p_vec_R += 0.5 * (1.0 - eig * dtz) * d_w[c] * col(Eigs.z.right, c);
                    } else {
                        p_vec_L += 0.5 * (-1.0 - eig * dtz) * d_w[c] * col(Eigs.z.right, c);
                    }
                }

                for (int c = 0; c < numvar-1; ++c) {
                    for (int q = 0; q < view.quad; ++q) {
                        view.fzR(c, q, k,   j, i) = p_vec_L[c];
                        view.fzL(c, q, k+1, j, i) = p_vec_R[c];
                    }
                }
            }
        }
    );
}

template<limiter TVD, int dim, class Sim>
AETHER_INLINE void PPM_sweep(Sim& sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;

    auto view = sim.view();
    auto prim = view.prim;

    const double dtx = view.dt / view.dx;
    [[maybe_unused]] const double dty_p = view.dt / view.dy;
    [[maybe_unused]] const double dtz_p = view.dt / view.dz;

    Kokkos::parallel_for(
        "PPM_sweep",
        loops::solver_sweep_policy(sim),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            vec p_vec, p_vec_L, p_vec_R, p_vec_L2, p_vec_R2;
            prims p;
            chars Eigs;
            double dty = dty_p;
            double dtz = dtz_p;

            for (int c = 0; c < numvar-1; ++c) {
                p_vec[c] = prim(c, k, j, i);
            }

            p.rho = p_vec[0];
            p.vx  = p_vec[1];
            if constexpr (P::HAS_VY) p.vy = p_vec[P::VY];
            if constexpr (P::HAS_VZ) p.vz = p_vec[P::VZ];
            p.p = p_vec[P::P];

            fill_eigenvectors(p, Eigs, view.gamma);

            vec d_w_i, d_w_ir, d_w_il;
            vec a0L, a0R;

            // x-sweep
            for (int c = 0; c < numvar-1; ++c) {
                p_vec_L[c]  = prim(c, k, j, i - 1);
                p_vec_L2[c] = prim(c, k, j, i - 2);
                p_vec_R[c]  = prim(c, k, j, i + 1);
                p_vec_R2[c] = prim(c, k, j, i + 2);
            }

            if constexpr (TVD == limiter::minmod) {
                d_w_i  = minmod(Eigs.x.left * (p_vec   - p_vec_L),
                                Eigs.x.left * (p_vec_R - p_vec));
                d_w_ir = minmod(Eigs.x.left * (p_vec_R  - p_vec),
                                Eigs.x.left * (p_vec_R2 - p_vec_R));
                d_w_il = minmod(Eigs.x.left * (p_vec_L  - p_vec_L2),
                                Eigs.x.left * (p_vec    - p_vec_L));
            } else if constexpr (TVD == limiter::mc) {
                d_w_i  = mc(0.5 * Eigs.x.left * (p_vec_R  - p_vec_L),
                            2.0 * Eigs.x.left * (p_vec_R  - p_vec),
                            2.0 * Eigs.x.left * (p_vec    - p_vec_L));
                d_w_ir = mc(0.5 * Eigs.x.left * (p_vec_R2 - p_vec),
                            2.0 * Eigs.x.left * (p_vec_R2 - p_vec_R),
                            2.0 * Eigs.x.left * (p_vec_R  - p_vec));
                d_w_il = mc(0.5 * Eigs.x.left * (p_vec    - p_vec_L2),
                            2.0 * Eigs.x.left * (p_vec    - p_vec_L),
                            2.0 * Eigs.x.left * (p_vec_L  - p_vec_L2));
            } else if constexpr (TVD == limiter::vanleer) {
                d_w_i  = van_leer(Eigs.x.left * (p_vec   - p_vec_L),
                                  Eigs.x.left * (p_vec_R - p_vec));
                d_w_ir = van_leer(Eigs.x.left * (p_vec_R  - p_vec),
                                  Eigs.x.left * (p_vec_R2 - p_vec_R));
                d_w_il = van_leer(Eigs.x.left * (p_vec_L  - p_vec_L2),
                                  Eigs.x.left * (p_vec    - p_vec_L));
            }

            d_w_i  = Eigs.x.right * d_w_i;
            d_w_ir = Eigs.x.right * d_w_ir;
            d_w_il = Eigs.x.right * d_w_il;

            a0L = 0.5 * (p_vec_L + p_vec  ) - (1.0 / 6.0) * (d_w_i  - d_w_il);
            a0R = 0.5 * (p_vec   + p_vec_R) - (1.0 / 6.0) * (d_w_ir - d_w_i);

            d_w_ir = (a0R - p_vec);
            d_w_il = (p_vec - a0L);
            p_vec_L = (a0R - a0L);
            p_vec_R = (a0R + a0L);

            for (int c = 0; c < numvar-1; ++c) {
                if (d_w_ir[c] * d_w_il[c] <= 0.0) {
                    a0L[c] = p_vec[c];
                    a0R[c] = p_vec[c];
                } else if (-p_vec_L[c] * p_vec_L[c] > 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                    a0R[c] = 3.0 * p_vec[c] - 2.0 * a0L[c];
                } else if (p_vec_L[c] * p_vec_L[c] < 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                    a0L[c] = 3.0 * p_vec[c] - 2.0 * a0R[c];
                }
            }

            vec C2 = 6.0 * (0.5 * (a0R + a0L) - p_vec);
            vec C1 = (a0R - a0L);
            vec C0 = p_vec - (1.0 / 12.0) * C2;

            vec delta_c1 = Eigs.x.left * C1;
            vec delta_c2 = Eigs.x.left * C2;

            p_vec_L = C0;
            p_vec_R = C0;

            for (int c = 0; c < numvar-1; ++c) {
                const double eig = Eigs.x.lambda(c);
                if (eig >= 0.0) {
                    p_vec_R += 0.5 * (1.0 - eig * dtx) * col(Eigs.x.right, c) * delta_c1[c]
                             + 0.25 * (1.0 - 2.0 * eig * dtx + (4.0 / 3.0) * (eig * dtx) * (eig * dtx))
                             * col(Eigs.x.right, c) * delta_c2[c];
                } else {
                    p_vec_L += 0.5 * (-1.0 - eig * dtx) * col(Eigs.x.right, c) * delta_c1[c]
                             + 0.25 * (1.0 + 2.0 * eig * dtx + (4.0 / 3.0) * (eig * dtx) * (eig * dtx))
                             * col(Eigs.x.right, c) * delta_c2[c];
                }
            }

            for (int c = 0; c < numvar-1; ++c) {
                for (int q = 0; q < view.quad; ++q) {
                    view.fxR(c, q, k, j, i)   = p_vec_L[c];
                    view.fxL(c, q, k, j, i+1) = p_vec_R[c];
                }
            }

            // y-sweep
            if constexpr (dim > 1) {
                for (int c = 0; c < numvar-1; ++c) {
                    p_vec_L[c]  = prim(c, k, j - 1, i);
                    p_vec_L2[c] = prim(c, k, j - 2, i);
                    p_vec_R[c]  = prim(c, k, j + 1, i);
                    p_vec_R2[c] = prim(c, k, j + 2, i);
                }

                if constexpr (TVD == limiter::minmod) {
                    d_w_i  = minmod(Eigs.y.left * (p_vec   - p_vec_L),
                                    Eigs.y.left * (p_vec_R - p_vec));
                    d_w_ir = minmod(Eigs.y.left * (p_vec_R  - p_vec),
                                    Eigs.y.left * (p_vec_R2 - p_vec_R));
                    d_w_il = minmod(Eigs.y.left * (p_vec_L  - p_vec_L2),
                                    Eigs.y.left * (p_vec    - p_vec_L));
                } else if constexpr (TVD == limiter::mc) {
                    d_w_i  = mc(0.5 * Eigs.y.left * (p_vec_R  - p_vec_L),
                                2.0 * Eigs.y.left * (p_vec_R  - p_vec),
                                2.0 * Eigs.y.left * (p_vec    - p_vec_L));
                    d_w_ir = mc(0.5 * Eigs.y.left * (p_vec_R2 - p_vec),
                                2.0 * Eigs.y.left * (p_vec_R2 - p_vec_R),
                                2.0 * Eigs.y.left * (p_vec_R  - p_vec));
                    d_w_il = mc(0.5 * Eigs.y.left * (p_vec    - p_vec_L2),
                                2.0 * Eigs.y.left * (p_vec    - p_vec_L),
                                2.0 * Eigs.y.left * (p_vec_L  - p_vec_L2));
                } else if constexpr (TVD == limiter::vanleer) {
                    d_w_i  = van_leer(Eigs.y.left * (p_vec   - p_vec_L),
                                      Eigs.y.left * (p_vec_R - p_vec));
                    d_w_ir = van_leer(Eigs.y.left * (p_vec_R  - p_vec),
                                      Eigs.y.left * (p_vec_R2 - p_vec_R));
                    d_w_il = van_leer(Eigs.y.left * (p_vec_L  - p_vec_L2),
                                      Eigs.y.left * (p_vec    - p_vec_L));
                }

                d_w_i  = Eigs.y.right * d_w_i;
                d_w_ir = Eigs.y.right * d_w_ir;
                d_w_il = Eigs.y.right * d_w_il;

                a0L = 0.5 * (p_vec_L + p_vec  ) - (1.0 / 6.0) * (d_w_i  - d_w_il);
                a0R = 0.5 * (p_vec   + p_vec_R) - (1.0 / 6.0) * (d_w_ir - d_w_i);

                d_w_ir = (a0R - p_vec);
                d_w_il = (p_vec - a0L);
                p_vec_L = (a0R - a0L);
                p_vec_R = (a0R + a0L);

                for (int c = 0; c < numvar-1; ++c) {
                    if (d_w_ir[c] * d_w_il[c] <= 0.0) {
                        a0L[c] = p_vec[c];
                        a0R[c] = p_vec[c];
                    } else if (-p_vec_L[c] * p_vec_L[c] > 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                        a0R[c] = 3.0 * p_vec[c] - 2.0 * a0L[c];
                    } else if (p_vec_L[c] * p_vec_L[c] < 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                        a0L[c] = 3.0 * p_vec[c] - 2.0 * a0R[c];
                    }
                }

                C2 = 6.0 * (0.5 * (a0R + a0L) - p_vec);
                C1 = (a0R - a0L);
                C0 = p_vec - (1.0 / 12.0) * C2;

                delta_c1 = Eigs.y.left * C1;
                delta_c2 = Eigs.y.left * C2;

                p_vec_L = C0;
                p_vec_R = C0;

                for (int c = 0; c < numvar-1; ++c) {
                    const double eig = Eigs.y.lambda(c);
                    if (eig >= 0.0) {
                        p_vec_R += 0.5 * (1.0 - eig * dty) * col(Eigs.y.right, c) * delta_c1[c]
                                 + 0.25 * (1.0 - 2.0 * eig * dty + (4.0 / 3.0) * (eig * dty) * (eig * dty))
                                 * col(Eigs.y.right, c) * delta_c2[c];
                    } else {
                        p_vec_L += 0.5 * (-1.0 - eig * dty) * col(Eigs.y.right, c) * delta_c1[c]
                                 + 0.25 * (1.0 + 2.0 * eig * dty + (4.0 / 3.0) * (eig * dty) * (eig * dty))
                                 * col(Eigs.y.right, c) * delta_c2[c];
                    }
                }

                for (int c = 0; c < numvar-1; ++c) {
                    for (int q = 0; q < view.quad; ++q) {
                        view.fyR(c, q, k, j,   i) = p_vec_L[c];
                        view.fyL(c, q, k, j+1, i) = p_vec_R[c];
                    }
                }
            }

            // z-sweep
            if constexpr (dim > 2) {
                for (int c = 0; c < numvar-1; ++c) {
                    p_vec_L[c]  = prim(c, k - 1, j, i);
                    p_vec_L2[c] = prim(c, k - 2, j, i);
                    p_vec_R[c]  = prim(c, k + 1, j, i);
                    p_vec_R2[c] = prim(c, k + 2, j, i);
                }

                if constexpr (TVD == limiter::minmod) {
                    d_w_i  = minmod(Eigs.z.left * (p_vec   - p_vec_L),
                                    Eigs.z.left * (p_vec_R - p_vec));
                    d_w_ir = minmod(Eigs.z.left * (p_vec_R  - p_vec),
                                    Eigs.z.left * (p_vec_R2 - p_vec_R));
                    d_w_il = minmod(Eigs.z.left * (p_vec_L  - p_vec_L2),
                                    Eigs.z.left * (p_vec    - p_vec_L));
                } else if constexpr (TVD == limiter::mc) {
                    d_w_i  = mc(0.5 * Eigs.z.left * (p_vec_R  - p_vec_L),
                                2.0 * Eigs.z.left * (p_vec_R  - p_vec),
                                2.0 * Eigs.z.left * (p_vec    - p_vec_L));
                    d_w_ir = mc(0.5 * Eigs.z.left * (p_vec_R2 - p_vec),
                                2.0 * Eigs.z.left * (p_vec_R2 - p_vec_R),
                                2.0 * Eigs.z.left * (p_vec_R  - p_vec));
                    d_w_il = mc(0.5 * Eigs.z.left * (p_vec    - p_vec_L2),
                                2.0 * Eigs.z.left * (p_vec    - p_vec_L),
                                2.0 * Eigs.z.left * (p_vec_L  - p_vec_L2));
                } else if constexpr (TVD == limiter::vanleer) {
                    d_w_i  = van_leer(Eigs.z.left * (p_vec   - p_vec_L),
                                      Eigs.z.left * (p_vec_R - p_vec));
                    d_w_ir = van_leer(Eigs.z.left * (p_vec_R  - p_vec),
                                      Eigs.z.left * (p_vec_R2 - p_vec_R));
                    d_w_il = van_leer(Eigs.z.left * (p_vec_L  - p_vec_L2),
                                      Eigs.z.left * (p_vec    - p_vec_L));
                }

                d_w_i  = Eigs.z.right * d_w_i;
                d_w_ir = Eigs.z.right * d_w_ir;
                d_w_il = Eigs.z.right * d_w_il;

                a0L = 0.5 * (p_vec_L + p_vec  ) - (1.0 / 6.0) * (d_w_i  - d_w_il);
                a0R = 0.5 * (p_vec   + p_vec_R) - (1.0 / 6.0) * (d_w_ir - d_w_i);

                d_w_ir = (a0R - p_vec);
                d_w_il = (p_vec - a0L);
                p_vec_L = (a0R - a0L);
                p_vec_R = (a0R + a0L);

                for (int c = 0; c < numvar-1; ++c) {
                    if (d_w_ir[c] * d_w_il[c] <= 0.0) {
                        a0L[c] = p_vec[c];
                        a0R[c] = p_vec[c];
                    } else if (-p_vec_L[c] * p_vec_L[c] > 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                        a0R[c] = 3.0 * p_vec[c] - 2.0 * a0L[c];
                    } else if (p_vec_L[c] * p_vec_L[c] < 6.0 * p_vec_L[c] * (p_vec[c] - 0.5 * p_vec_R[c])) {
                        a0L[c] = 3.0 * p_vec[c] - 2.0 * a0R[c];
                    }
                }

                C2 = 6.0 * (0.5 * (a0R + a0L) - p_vec);
                C1 = (a0R - a0L);
                C0 = p_vec - (1.0 / 12.0) * C2;

                delta_c1 = Eigs.z.left * C1;
                delta_c2 = Eigs.z.left * C2;

                p_vec_L = C0;
                p_vec_R = C0;

                for (int c = 0; c < numvar-1; ++c) {
                    const double eig = Eigs.z.lambda(c);
                    if (eig >= 0.0) {
                        p_vec_R += 0.5 * (1.0 - eig * dtz) * col(Eigs.z.right, c) * delta_c1[c]
                                 + 0.25 * (1.0 - 2.0 * eig * dtz + (4.0 / 3.0) * (eig * dtz) * (eig * dtz))
                                 * col(Eigs.z.right, c) * delta_c2[c];
                    } else {
                        p_vec_L += 0.5 * (-1.0 - eig * dtz) * col(Eigs.z.right, c) * delta_c1[c]
                                 + 0.25 * (1.0 + 2.0 * eig * dtz + (4.0 / 3.0) * (eig * dtz) * (eig * dtz))
                                 * col(Eigs.z.right, c) * delta_c2[c];
                    }
                }

                for (int c = 0; c < numvar-1; ++c) {
                    for (int q = 0; q < view.quad; ++q) {
                        view.fzR(c, q, k,   j, i) = p_vec_L[c];
                        view.fzL(c, q, k+1, j, i) = p_vec_R[c];
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
            if (Sim.cfg.slope_limiter == limiter::minmod) {
                PLM_sweep<limiter::minmod, AETHER_DIM>(Sim);
            } else if (Sim.cfg.slope_limiter == limiter::mc) {
                PLM_sweep<limiter::mc, AETHER_DIM>(Sim);
            } else if (Sim.cfg.slope_limiter == limiter::vanleer) {
                PLM_sweep<limiter::vanleer, AETHER_DIM>(Sim);
            } else {
                throw std::runtime_error("Slope limiter type not available in PLM");
            }
            break;

        case solver::ppm:
            if (Sim.cfg.slope_limiter == limiter::minmod) {
                PPM_sweep<limiter::minmod, AETHER_DIM>(Sim);
            } else if (Sim.cfg.slope_limiter == limiter::mc) {
                PPM_sweep<limiter::mc, AETHER_DIM>(Sim);
            } else if (Sim.cfg.slope_limiter == limiter::vanleer) {
                PPM_sweep<limiter::vanleer, AETHER_DIM>(Sim);
            } else {
                throw std::runtime_error("Slope limiter type not available in PPM");
            }
            break;

        default:
            throw std::runtime_error("Space_solve: unknown space solver");
    }
}

} // namespace aether::core
