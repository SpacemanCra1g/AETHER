#include <Kokkos_Core.hpp>
#include <cmath>
#include <aether/core/CTU/ctu_half_time_step.hpp>
#include <aether/core/RiemannDispatch.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/con_layout.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/physics/api.hpp>

namespace aether::core {

using prims = aether::phys::prims;
using cons  = aether::phys::cons;
using C     = aether::con::Cons;
using P     = aether::prim::Prim;

struct idx{
    int q=0, k=0, j=0, i=0;
};

namespace detail {

// ============================================================
// face primitive helpers
// ============================================================

template<class FaceT>
KOKKOS_INLINE_FUNCTION
prims load_face_prims(const FaceT& faces, idx index) noexcept {
    prims x{};
    x.rho = faces(P::RHO, index.q, index.k, index.j, index.i);
    x.vx  = faces(P::VX,  index.q, index.k, index.j, index.i);
    x.vy = faces(P::VY, index.q, index.k, index.j, index.i);
    x.vz = (P::HAS_VZ) ? faces(P::VZ, index.q, index.k, index.j, index.i) : 0.0;
    x.p   = faces(P::P, index.q, index.k, index.j, index.i);
    return x;
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
void store_face_prims(const FaceT& faces, idx index,
                      const prims& x) noexcept {
    faces(P::RHO, index.q, index.k, index.j, index.i) = x.rho;
    faces(P::VX,  index.q, index.k, index.j, index.i) = x.vx;
    faces(P::VY, index.q, index.k, index.j, index.i) = x.vy;
    if constexpr (P::HAS_VZ) faces(P::VZ, index.q, index.k, index.j, index.i) = x.vz;
    faces(P::P, index.q, index.k, index.j, index.i) = x.p;
}

KOKKOS_INLINE_FUNCTION
bool prims_valid(const prims& x) noexcept {
    return std::isfinite(x.rho) && std::isfinite(x.p) && (x.rho > 0.0) && (x.p > 0.0);
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
void backup_cons(FaceT& backup, idx index,
                      const cons& x) noexcept{

    backup(C::RHO,index.q,index.k,index.j,index.i) = x.rho;
    backup(C::MX,index.q,index.k,index.j,index.i) = x.mx;
    backup(C::MY,index.q,index.k,index.j,index.i) = x.my;
    if constexpr (C::HAS_MZ) backup(C::MZ,index.q,index.k,index.j,index.i) = x.mz;
    backup(C::E,index.q,index.k,index.j,index.i) = x.E;
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
cons load_backup_cons(FaceT& backup, idx index) noexcept{
    cons x{};
    x.rho = backup(C::RHO,index.q,index.k,index.j,index.i);
    x.mx = backup(C::MX,index.q,index.k,index.j,index.i);
    x.my = backup(C::MY,index.q,index.k,index.j,index.i);
    x.mz = (C::HAS_MZ) ? backup(C::MZ,index.q,index.k,index.j,index.i) : 0.0;
    x.E = backup(C::E,index.q,index.k,index.j,index.i);
    return x;
}

// ============================================================
// conservative flux difference helper
// ============================================================

template<class FluxT>
KOKKOS_INLINE_FUNCTION
cons add_flux_diff(const cons& base,
                   const FluxT& F, idx idx_lo,
                   idx idx_hi, const double coeff) noexcept {
    cons x = base;
    x.rho += coeff * (F(C::RHO, idx_hi.q, idx_hi.k, idx_hi.j, idx_hi.i) - F(C::RHO, idx_lo.q, idx_lo.k, idx_lo.j, idx_lo.i));
    x.mx += coeff * (F(C::MX, idx_hi.q, idx_hi.k, idx_hi.j, idx_hi.i) - F(C::MX, idx_lo.q, idx_lo.k, idx_lo.j, idx_lo.i));
    x.my += coeff * (F(C::MY, idx_hi.q, idx_hi.k, idx_hi.j, idx_hi.i) - F(C::MY, idx_lo.q, idx_lo.k, idx_lo.j, idx_lo.i));
    if constexpr (C::HAS_MZ) {
        x.mz += coeff * (F(C::MZ, idx_hi.q, idx_hi.k, idx_hi.j, idx_hi.i) - F(C::MZ, idx_lo.q, idx_lo.k, idx_lo.j, idx_lo.i));
    }
    x.E += coeff * (F(C::E, idx_hi.q, idx_hi.k, idx_hi.j, idx_hi.i) - F(C::E, idx_lo.q, idx_lo.k, idx_lo.j, idx_lo.i));
    return x;
}

// ============================================================
// primitive recovery helpers
// ============================================================

KOKKOS_INLINE_FUNCTION
prims recover_prims(cons x, const double gamma) noexcept {
    return aether::phys::cons_to_prims_cell(x, gamma);
}

KOKKOS_INLINE_FUNCTION
prims recover_prims_or_fallback(cons corrected,
                                     cons fallback,
                                     double gamma) noexcept {
    prims qp = recover_prims(corrected, gamma);
    if (!prims_valid(qp)) {
        qp = recover_prims(fallback, gamma);
    }
    return qp;
}


// ============================================================
// common wrappers
// ============================================================

template<class FaceOutT, class FluxT>
KOKKOS_INLINE_FUNCTION
void correct_face_2d(FaceOutT& out_faces, idx idx_o,
                     FluxT& transverse_flux,
                     idx idx_lo, idx idx_hi,
                     double coeff,
                     double gamma) noexcept {
    const prims q0p = load_face_prims(out_faces, idx_o);
    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
    const cons  qc  = add_flux_diff(q0, transverse_flux, idx_lo, idx_hi, coeff);
    const prims qp  = recover_prims(qc, gamma);
    store_face_prims(out_faces, idx_o, qp);
}

template<class FaceOutT, class BackupT, class Flux1T, class Flux2T>
KOKKOS_INLINE_FUNCTION
void write_half_step_with_fallback(FaceOutT& out_faces,
                                   idx idx_o,
                                   BackupT& backup,
                                   Flux1T& F1,
                                   idx f1_lo, idx f1_hi,
                                   const double c1,
                                   Flux2T& F2,
                                   idx f2_lo, idx f2_hi,
                                   const double c2,
                                   const double gamma) noexcept {

    cons q0 = load_backup_cons(backup, idx_o);
    cons qc = q0;

    qc = add_flux_diff(qc, F1, f1_lo, f1_hi, c1);
    qc = add_flux_diff(qc, F2, f2_lo, f2_hi, c2);

    prims qp = recover_prims_or_fallback(qc,q0, gamma);
    store_face_prims(out_faces, idx_o, qp);
}
}

#if AETHER_DIM > 1
template<>
void ctu_half_time_correction<2>(Simulation& sim, Simulation::View view) {

    using exec_space = typename Simulation::policy_type::execution_space;
    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int jb = sim.cells.jbegin();
    const int je = sim.cells.jend();

    const double gamma    = sim.grid.gamma;
    const double dxt_half = 0.5 * sim.time.dt / sim.grid.dx;
    const double dyt_half = 0.5 * sim.time.dt / sim.grid.dy;
    const int quad        = sim.grid.quad;

    auto Lx = view.fxL;
    auto Rx = view.fxR;
    auto Fx = view.fx;

    auto Ly = view.fyL;
    auto Ry = view.fyR;
    auto Fy = view.fy;

    Kokkos::parallel_for(
        "ctu_half_time_correction_2d",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {jb - 1, ib - 1},
            {je + 1, ie + 1}
        ),
        KOKKOS_LAMBDA(const int j, const int i) {
            for (int q = 0; q < quad; ++q) {
                idx x_right(q, 0, j  , i+1), x_left(  q, 0, j, i);
                idx y_top(  q, 0, j+1, i  ), y_bottom(q, 0, j, i);
                
                detail::correct_face_2d(Lx, x_right, Fy, y_bottom, y_top, -dyt_half, gamma);
                detail::correct_face_2d(Rx, x_left,  Fy, y_bottom, y_top, -dyt_half, gamma);
                detail::correct_face_2d(Ly, y_top,   Fx, x_left,   x_right, -dxt_half, gamma);
                detail::correct_face_2d(Ry, y_bottom,Fx, x_left,   x_right, -dxt_half, gamma);
            }
        }
    );
}
#endif


// ============================================================
// 3D CTU half-step correction
// ============================================================

#if AETHER_DIM > 2
template<>
void ctu_half_time_correction<3>(Simulation& sim, Simulation::View view) {
    using exec_space = typename Simulation::policy_type::execution_space;

    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int jb = sim.cells.jbegin();
    const int je = sim.cells.jend();
    const int kb = sim.cells.kbegin();
    const int ke = sim.cells.kend();

    const double gamma     = sim.grid.gamma;
    const double dxt_third = sim.time.dt / (3.0 * sim.grid.dx);
    const double dyt_third = sim.time.dt / (3.0 * sim.grid.dy);
    const double dzt_third = sim.time.dt / (3.0 * sim.grid.dz);

    const double dxt_half = sim.time.dt / (2.0 * sim.grid.dx);
    const double dyt_half = sim.time.dt / (2.0 * sim.grid.dy);
    const double dzt_half = sim.time.dt / (2.0 * sim.grid.dz);

    const int quad = sim.grid.quad;

    auto ctu = sim.ctu_view();

    auto Lx = view.fxL;  auto Rx = view.fxR;  auto Fx = view.fx;
    auto Ly = view.fyL;  auto Ry = view.fyR;  auto Fy = view.fy;
    auto Lz = view.fzL;  auto Rz = view.fzR;  auto Fz = view.fz;

    auto ctu_Lx = ctu.fxL;  auto ctu_Rx = ctu.fxR;  auto ctu_Fx = ctu.fx;
    auto ctu_Ly = ctu.fyL;  auto ctu_Ry = ctu.fyR;  auto ctu_Fy = ctu.fy;
    auto ctu_Lz = ctu.fzL;  auto ctu_Rz = ctu.fzR;  auto ctu_Fz = ctu.fz;

    auto xL_bak = ctu.xL_bak, xR_bak = ctu.xR_bak;
    auto yL_bak = ctu.yL_bak, yR_bak = ctu.yR_bak;
    auto zL_bak = ctu.zL_bak, zR_bak = ctu.zR_bak;

    Kokkos::parallel_for(
        "ctu_predictor_stage_3d",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {kb - 1, jb - 1, ib - 1},
            {ke + 1, je + 1, ie + 1}
        ),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int q = 0; q < quad; ++q) {
                
                idx x_right(q, k  , j  , i+1), x_left(  q, k, j, i);
                idx y_top(  q, k  , j+1, i)  , y_bottom(q, k, j, i);
                idx z_up(   q, k+1, j  , i)  , z_down(  q, k, j, i);

                {
                    prims q0p = detail::load_face_prims(Lx, x_right);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(xL_bak, x_right, q0);

                    detail::store_face_prims(
                        Lx, x_right,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fy, y_bottom, y_top, -dyt_third), gamma));

                    detail::store_face_prims(
                        ctu_Lx, x_right,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fz, z_down, z_up, -dzt_third), gamma));
                }

                {
                    prims q0p = detail::load_face_prims(Rx, x_left);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(xR_bak, x_left, q0);

                    detail::store_face_prims(
                        Rx, x_left,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fy, y_bottom, y_top, -dyt_third), gamma));

                    detail::store_face_prims(
                        ctu_Rx, x_left,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fz, z_down, z_up, -dzt_third), gamma));
                }

                {
                    prims q0p = detail::load_face_prims(Ly, y_top);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(yL_bak, y_top, q0);

                    detail::store_face_prims(
                        Ly, y_top,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fx, x_left, x_right, -dxt_third), gamma));

                    detail::store_face_prims(
                        ctu_Ly, y_top,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fz, z_down, z_up, -dzt_third), gamma));
                }

                {
                    prims q0p = detail::load_face_prims(Ry, y_bottom);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(yR_bak, y_bottom, q0);

                    detail::store_face_prims(
                        Ry, y_bottom,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fx, x_left, x_right, -dxt_third), gamma));

                    detail::store_face_prims(
                        ctu_Ry, y_bottom,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fz, z_down, z_up, -dzt_third), gamma));
                }

                {
                    prims q0p = detail::load_face_prims(Lz, z_up);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(zL_bak, z_up, q0);

                    detail::store_face_prims(
                        Lz, z_up,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fx, x_left, x_right, -dxt_third), gamma));

                    detail::store_face_prims(
                        ctu_Lz, z_up,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fy, y_bottom, y_top, -dyt_third), gamma));
                }

                {
                    prims q0p = detail::load_face_prims(Rz, z_down);
                    cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(zR_bak, z_down, q0);

                    detail::store_face_prims(
                        Rz, z_down,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fx, x_left, x_right, -dxt_third), gamma));

                    detail::store_face_prims(
                        ctu_Rz, z_down,
                        detail::recover_prims(
                            detail::add_flux_diff(q0, Fy, y_bottom, y_top, -dyt_third), gamma));
                }
            }
        }
    );
    Riemann_dispatch(sim, view);
    Riemann_dispatch(sim, ctu);

    Kokkos::parallel_for(
        "ctu_predictor_stage1_3d",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {kb - 1, jb - 1, ib - 1},
            {ke + 1, je + 1, ie + 1}
        ),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            
            for (int q = 0; q < quad; q++){

                idx x_right(q, k  , j  , i+1), x_left(  q, k, j, i);
                idx y_top(  q, k  , j+1, i)  , y_bottom(q, k, j, i);
                idx z_up(   q, k+1, j  , i)  , z_down(  q, k, j, i);

                detail::write_half_step_with_fallback(
                    Lx, x_right, xL_bak,
                    ctu_Fy, y_top, y_bottom, +dyt_half,
                    ctu_Fz, z_up,  z_down,   +dzt_half,
                    gamma);

                detail::write_half_step_with_fallback(
                    Rx, x_left, xR_bak,
                    ctu_Fy, y_top, y_bottom, +dyt_half,
                    ctu_Fz, z_up,  z_down,   +dzt_half,
                    gamma);

                detail::write_half_step_with_fallback(
                    Ly, y_top, yL_bak,
                    ctu_Fx, x_right, x_left, +dxt_half,
                    Fz,     z_up,    z_down, +dzt_half,
                    gamma);

                detail::write_half_step_with_fallback(
                    Ry, y_bottom, yR_bak,
                    ctu_Fx, x_right, x_left, +dxt_half,
                    Fz,     z_up,    z_down, +dzt_half,
                    gamma);

                detail::write_half_step_with_fallback(
                    Lz, z_up, zL_bak,
                    Fx, x_right, x_left, +dxt_half,
                    Fy, y_top,   y_bottom, +dyt_half,
                    gamma);

                detail::write_half_step_with_fallback(
                    Rz, z_down, zR_bak,
                    Fx, x_right, x_left, +dxt_half,
                    Fy, y_top,   y_bottom, +dyt_half,
                    gamma);

            }
        }
    );
}
#endif

} // namespace aether::core