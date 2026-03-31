#include "Kokkos_Macros.hpp"
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

namespace detail {

// ============================================================
// face primitive helpers
// ============================================================

template<class FaceT>
KOKKOS_INLINE_FUNCTION
prims load_face_prims(const FaceT& faces, const int q,
                      const int k, const int j, const int i) noexcept {
    prims x{};
    x.rho = faces(P::RHO, q, k, j, i);
    x.vx  = faces(P::VX,  q, k, j, i);
    x.vy  = 0.0;
    x.vz  = 0.0;
    x.vy = faces(P::VY, q, k, j, i);
    x.vz = (P::HAS_VZ) ? faces(P::VZ, q, k, j, i) : 0.0;
    x.p   = faces(P::P, q, k, j, i);
    return x;
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
void store_face_prims(const FaceT& faces, const int q,
                      const int k, const int j, const int i,
                      const prims& x) noexcept {
    faces(P::RHO, q, k, j, i) = x.rho;
    faces(P::VX,  q, k, j, i) = x.vx;
    faces(P::VY, q, k, j, i) = x.vy;
    if constexpr (P::HAS_VZ) faces(P::VZ, q, k, j, i) = x.vz;
    faces(P::P,   q, k, j, i) = x.p;
}

KOKKOS_INLINE_FUNCTION
bool prims_valid(const prims& x) noexcept {
    return std::isfinite(x.rho) && std::isfinite(x.p) && (x.rho > 0.0) && (x.p > 0.0);
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
void backup_cons(FaceT& backup, const int q,
                      const int k, const int j, const int i,
                      const cons& x) noexcept{

    backup(C::RHO,q,k,j,i) = x.rho;
    backup(C::MX,q,k,j,i) = x.mx;
    backup(C::MY,q,k,j,i) = x.my;
    if constexpr (C::HAS_MZ) backup(C::MZ,q,k,j,i) = x.mz;
    backup(C::E,q,k,j,i) = x.E;
}

template<class FaceT>
KOKKOS_INLINE_FUNCTION
void load_backup_cons(FaceT& backup, const int q,
                      const int k, const int j, const int i) noexcept{
    cons x{};
    x.rho = backup(C::RHO,q,k,j,i);
    x.mx = backup(C::MX,q,k,j,i);
    x.my = backup(C::MY,q,k,j,i);
    x.mz = (C::HAS_MZ) ? backup(C::MZ,q,k,j,i) : 0.0;
    x.E = backup(C::E,q,k,j,i);
}

// ============================================================
// conservative flux difference helper
// ============================================================

template<class FluxT>
KOKKOS_INLINE_FUNCTION
cons add_flux_diff(const cons& base,
                   const FluxT& F, const int q,
                   const int klo, const int jlo, const int ilo,
                   const int khi, const int jhi, const int ihi,
                   const double coeff) noexcept {
    cons x = base;
    x.rho += coeff * (F(C::RHO, q, khi, jhi, ihi) - F(C::RHO, q, klo, jlo, ilo));
    x.mx  += coeff * (F(C::MX,  q, khi, jhi, ihi) - F(C::MX,  q, klo, jlo, ilo));
    x.my  += coeff * (F(C::MY,  q, khi, jhi, ihi) - F(C::MY,  q, klo, jlo, ilo));
    if constexpr (C::HAS_MZ) {
        x.mz += coeff * (F(C::MZ, q, khi, jhi, ihi) - F(C::MZ, q, klo, jlo, ilo));
    }
    x.E   += coeff * (F(C::E,   q, khi, jhi, ihi) - F(C::E,   q, klo, jlo, ilo));
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
void correct_face_2d(FaceOutT& out_faces, const int q,
                     const int ok, const int oj, const int oi,
                     FluxT& transverse_flux,
                     int klo, const int jlo, const int ilo,
                     int khi, const int jhi, const int ihi,
                     double coeff,
                     double gamma) noexcept {
    const prims q0p = load_face_prims(out_faces, q, ok, oj, oi);
    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
    const cons  qc  = add_flux_diff(q0, transverse_flux, q,
                                    klo, jlo, ilo,
                                    khi, jhi, ihi,
                                    coeff);
    const prims qp  = recover_prims(qc, gamma);
    store_face_prims(out_faces, q, ok, oj, oi, qp);
}

template<class FaceOutT, class BackupT, class Flux1T, class Flux2T>
KOKKOS_INLINE_FUNCTION
void write_half_step_with_fallback(FaceOutT& out_faces,
                                   const int q,
                                   const int ok, const int oj, const int oi,
                                   BackupT& backup,
                                   Flux1T& F1,
                                   const int f1_klo, const int f1_jlo, const int f1_ilo,
                                   const int f1_khi, const int f1_jhi, const int f1_ihi,
                                   const double c1,
                                   Flux2T& F2,
                                   const int f2_klo, const int f2_jlo, const int f2_ilo,
                                   const int f2_khi, const int f2_jhi, const int f2_ihi,
                                   const double c2,
                                   const double gamma) noexcept {

    cons q0 = load_backup_cons(backup, q, ok, oj, oi);
    cons qc = q0;

    qc = add_flux_diff(qc, F1, q, f1_klo, f1_jlo, f1_ilo, f1_khi, f1_jhi, f1_ihi, c1);
    qc = add_flux_diff(qc, F2, q, f2_klo, f2_jlo, f2_ilo, f2_khi, f2_jhi, f2_ihi, c2);

    prims qp = recover_prims_or_fallback(qc,q0, gamma);
    store_face_prims(out_faces, q, ok, oj, oi, qp);
}


#if AETHER_DIM > 1
template<>
void ctu_half_time_correction<2>(Simulation& sim, Simulation::View view) {

    using exec_space = typename Simulation::policy_type::execution_space;
    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int jb = sim.cells.jbegin();
    const int je = sim.cells.jend();

    constexpr int k = 0;

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
                
            detail::correct_face_2d(Lx,q, 0, j  , i+1, Fy, 0, j, i, 0, j+1, i  , -dyt_half, gamma);
            detail::correct_face_2d(Rx,q, 0, j  , i  , Fy, 0, j, i, 0, j+1, i  , -dyt_half, gamma);
            detail::correct_face_2d(Ly,q, 0, j+1, i  , Fx, 0, j, i, 0, j  , i+1, -dxt_half, gamma);
            detail::correct_face_2d(Ry,q, 0, j  , i  , Fx, 0, j, i, 0, j  , i+1, -dxt_half, gamma);
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

    const int quad = sim.grid.quad;

    auto ctu = sim.ctu_view();

    auto Lx = view.fxL;  auto Rx = view.fxR;  auto Fx = view.fx;
    auto Ly = view.fyL;  auto Ry = view.fyR;  auto Fy = view.fy;
    auto Lz = view.fzL;  auto Rz = view.fzR;  auto Fz = view.fz;

    auto ctu_Lx = ctu.fxL;  auto ctu_Rx = ctu.fxR;  auto ctu_Fx = ctu.fx;
    auto ctu_Ly = ctu.fyL;  auto ctu_Ry = ctu.fyR;  auto ctu_Fy = ctu.fy;
    auto ctu_Lz = ctu.fzL;  auto ctu_Rz = ctu.fzR;  auto ctu_Fz = ctu.fz;

    // Replace these with your actual backup-buffer members if their names differ.
    auto xL_bak = ctu.fxL;
    auto xR_bak = ctu.fxR;
    auto yL_bak = ctu.fyL;
    auto yR_bak = ctu.fyR;
    auto zL_bak = ctu.fzL;
    auto zR_bak = ctu.fzR;

    Kokkos::parallel_for(
        "ctu_predictor_stage1_3d",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {kb - 1, jb - 1, ib - 1},
            {ke + 1, je + 1, ie + 1}
        ),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int q = 0; q < quad; ++q) {

                {
                    const prims q0p = detail::load_face_prims(Lx, q, k, j, i + 1);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(xL_bak, q, k, j, i + 1, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fy, q,
                            k, j,     i,
                            k, j + 1, i,
                            +dyt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Lx, q, k, j, i + 1, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fz, q,
                            k,     j, i,
                            k + 1, j, i,
                            +dzt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Lx, q, k, j, i + 1, qp);
                    }
                }

                {
                    const prims q0p = detail::load_face_prims(Rx, q, k, j, i);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(xR_bak, q, k, j, i, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fy, q,
                            k, j,     i,
                            k, j + 1, i,
                            +dyt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Rx, q, k, j, i, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fz, q,
                            k,     j, i,
                            k + 1, j, i,
                            +dzt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Rx, q, k, j, i, qp);
                    }
                }

                {
                    const prims q0p = detail::load_face_prims(Ly, q, k, j + 1, i);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(yL_bak, q, k, j + 1, i, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fx, q,
                            k, j, i + 1,
                            k, j, i,
                            +dxt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Ly, q, k, j + 1, i, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fz, q,
                            k,     j, i,
                            k + 1, j, i,
                            +dzt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Ly, q, k, j + 1, i, qp);
                    }
                }

                {
                    const prims q0p = detail::load_face_prims(Ry, q, k, j, i);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(yR_bak, q, k, j, i, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fx, q,
                            k, j, i + 1,
                            k, j, i,
                            +dxt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Ry, q, k, j, i, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fz, q,
                            k,     j, i,
                            k + 1, j, i,
                            +dzt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Ry, q, k, j, i, qp);
                    }
                }

                {
                    const prims q0p = detail::load_face_prims(Lz, q, k + 1, j, i);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(zL_bak, q, k + 1, j, i, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fx, q,
                            k, j, i + 1,
                            k, j, i,
                            +dxt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Lz, q, k + 1, j, i, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fy, q,
                            k, j,     i,
                            k, j + 1, i,
                            +dyt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Lz, q, k + 1, j, i, qp);
                    }
                }

                {
                    const prims q0p = detail::load_face_prims(Rz, q, k, j, i);
                    const cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
                    detail::backup_cons(zR_bak, q, k, j, i, q0);

                    {
                        const cons qc = detail::add_flux_diff(q0, Fx, q,
                            k, j, i + 1,
                            k, j, i,
                            +dxt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(Rz, q, k, j, i, qp);
                    }

                    {
                        const cons qc = detail::add_flux_diff(q0, Fy, q,
                            k, j,     i,
                            k, j + 1, i,
                            +dyt_third);
                        const prims qp = detail::recover_prims_or_face_fallback(qc, q0p, gamma);
                        detail::store_face_prims(ctu_Rz, q, k, j, i, qp);
                    }
                }
            }
        }
    );
}

template<>
void ctu_total_correction<3>(Simulation& sim, Simulation::View view) {
    using exec_space = typename Simulation::policy_type::execution_space;

    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int jb = sim.cells.jbegin();
    const int je = sim.cells.jend();
    const int kb = sim.cells.kbegin();
    const int ke = sim.cells.kend();

    const double gamma    = sim.grid.gamma;
    const double dxt_half = 0.5 * sim.time.dt / sim.grid.dx;
    const double dyt_half = 0.5 * sim.time.dt / sim.grid.dy;
    const double dzt_half = 0.5 * sim.time.dt / sim.grid.dz;

    const int quad = sim.grid.quad;

    auto ctu = sim.ctu_view();

    auto Lx = view.fxL;  auto Rx = view.fxR;  auto Fx = view.fx;
    auto Ly = view.fyL;  auto Ry = view.fyR;  auto Fy = view.fy;
    auto Lz = view.fzL;  auto Rz = view.fzR;  auto Fz = view.fz;

    auto ctu_Fx = ctu.fx;
    auto ctu_Fy = ctu.fy;
    auto ctu_Fz = ctu.fz;

    // Replace these with your actual backup-buffer members if their names differ.
    auto xL_bak = ctu.fxL;
    auto xR_bak = ctu.fxR;
    auto yL_bak = ctu.fyL;
    auto yR_bak = ctu.fyR;
    auto zL_bak = ctu.fzL;
    auto zR_bak = ctu.fzR;

    Kokkos::parallel_for(
        "ctu_predictor_stage2_3d",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {kb - 1, jb - 1, ib - 1},
            {ke + 1, je + 1, ie + 1}
        ),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            for (int q = 0; q < quad; ++q) {

                detail::write_half_step_with_fallback(
                    Lx, q, k, j, i + 1,
                    xL_bak,
                    ctu_Fy,
                    k, j + 1, i,
                    k, j,     i,
                    +dyt_half,
                    ctu_Fz,
                    k + 1, j, i,
                    k,     j, i,
                    +dzt_half,
                    gamma
                );

                detail::write_half_step_with_fallback(
                    Rx, q, k, j, i,
                    xR_bak,
                    ctu_Fy,
                    k, j + 1, i,
                    k, j,     i,
                    +dyt_half,
                    ctu_Fz,
                    k + 1, j, i,
                    k,     j, i,
                    +dzt_half,
                    gamma
                );

                detail::write_half_step_with_fallback(
                    Ly, q, k, j + 1, i,
                    yL_bak,
                    ctu_Fx,
                    k, j, i + 1,
                    k, j, i,
                    +dxt_half,
                    Fz,
                    k + 1, j, i,
                    k,     j, i,
                    +dzt_half,
                    gamma
                );

                detail::write_half_step_with_fallback(
                    Ry, q, k, j, i,
                    yR_bak,
                    ctu_Fx,
                    k, j, i + 1,
                    k, j, i,
                    +dxt_half,
                    Fz,
                    k + 1, j, i,
                    k,     j, i,
                    +dzt_half,
                    gamma
                );

                detail::write_half_step_with_fallback(
                    Lz, q, k + 1, j, i,
                    zL_bak,
                    Fx,
                    k, j, i + 1,
                    k, j, i,
                    +dxt_half,
                    Fy,
                    k, j + 1, i,
                    k, j,     i,
                    +dyt_half,
                    gamma
                );

                detail::write_half_step_with_fallback(
                    Rz, q, k, j, i,
                    zR_bak,
                    Fx,
                    k, j, i + 1,
                    k, j, i,
                    +dxt_half,
                    Fy,
                    k, j + 1, i,
                    k, j,     i,
                    +dyt_half,
                    gamma
                );
            }
        }
    );
}
#endif

} // namespace aether::core