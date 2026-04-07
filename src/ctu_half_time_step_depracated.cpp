// #include <aether/core/CTU/ctu_half_time_step.hpp>
// #include <aether/physics/api.hpp>
// #include "aether/core/RiemannDispatch.hpp"
// #include "aether/physics/euler/convert.hpp"
// #include <aether/core/con_layout.hpp>
// #include <cmath>
// #include <cstddef>

// namespace aether::core {
// using prims = aether::phys::prims;
// using cons  = aether::phys::cons;
// using C     = aether::con::Cons;

// namespace detail {

// // ============================================================
// // basic face / backup utilities
// // ============================================================

// template<class FaceT>
// AETHER_INLINE prims load_face_prims(FaceT& faces, std::size_t idx) noexcept {
//     prims q{};
//     q.rho = faces.comp[P::RHO][idx];
//     q.vx  = faces.comp[P::VX][idx];
//     q.vy  = faces.comp[P::VY][idx];
//     q.vz  = (P::HAS_VZ) ? faces.comp[P::VZ][idx] : 0.0;
//     q.p   = faces.comp[P::P][idx];
//     return q;
// }

// template<class FaceT>
// AETHER_INLINE void store_face_prims(FaceT& faces, std::size_t idx, const prims& q) noexcept {
//     faces.comp[P::RHO][idx] = q.rho;
//     faces.comp[P::VX][idx]  = q.vx;
//     faces.comp[P::VY][idx]  = q.vy;
//     if constexpr (P::HAS_VZ) faces.comp[P::VZ][idx] = q.vz;
//     faces.comp[P::P][idx]   = q.p;
// }

// AETHER_INLINE bool prims_valid(const prims& q) noexcept {
//     return std::isfinite(q.rho) && std::isfinite(q.p) && (q.rho > 0.0) && (q.p > 0.0);
// }

// template<class FaceT>
// AETHER_INLINE void backup_cons(FaceT& backup, std::size_t idx, const cons& q) noexcept {
//     backup.comp[C::RHO][idx] = q.rho;
//     backup.comp[C::MX][idx]  = q.mx;
//     backup.comp[C::MY][idx]  = q.my;
//     if constexpr (C::HAS_MZ) backup.comp[C::MZ][idx] = q.mz;
//     backup.comp[C::E][idx]   = q.E;
// }

// template<class FaceT>
// AETHER_INLINE cons load_backup_cons(FaceT& backup, std::size_t idx) noexcept {
//     cons q{};
//     q.rho = backup.comp[C::RHO][idx];
//     q.mx  = backup.comp[C::MX][idx];
//     q.my  = backup.comp[C::MY][idx];
//     q.mz  = (C::HAS_MZ) ? backup.comp[C::MZ][idx] : 0.0;
//     q.E   = backup.comp[C::E][idx];
//     return q;
// }

// // ============================================================
// // conservative flux-difference helper
// // one generic routine replaces x/y/z variants
// // ============================================================

// template<class FluxT>
// AETHER_INLINE cons add_flux_diff(const cons& base,
//                                  FluxT& F,
//                                  std::size_t lo,
//                                  std::size_t hi,
//                                  double coeff) noexcept {
//     cons q = base;
//     q.rho += coeff * (F.comp[C::RHO][hi] - F.comp[C::RHO][lo]);
//     q.mx  += coeff * (F.comp[C::MX][hi]  - F.comp[C::MX][lo]);
//     q.my  += coeff * (F.comp[C::MY][hi]  - F.comp[C::MY][lo]);
//     if constexpr (C::HAS_MZ) {
//         q.mz += coeff * (F.comp[C::MZ][hi] - F.comp[C::MZ][lo]);
//     }
//     q.E   += coeff * (F.comp[C::E][hi]   - F.comp[C::E][lo]);
//     return q;
// }

// // ============================================================
// // primitive recovery helpers
// // ============================================================

// AETHER_INLINE prims recover_prims(cons q, double gamma) noexcept {
//     return aether::phys::cons_to_prims_cell(q, gamma);
// }

// AETHER_INLINE prims recover_prims_or_fallback(cons corrected,
//                                               cons fallback,
//                                               double gamma) noexcept {
//     prims q = aether::phys::cons_to_prims_cell(corrected, gamma);
//     if (!prims_valid(q)) {
//         q = aether::phys::cons_to_prims_cell(fallback, gamma);
//     }
//     return q;
// }

// // ============================================================
// // common wrappers
// // ============================================================

// template<class FaceOutT, class FluxT>
// AETHER_INLINE void correct_face_2d(FaceOutT& out_faces,
//                                    std::size_t out_idx,
//                                    FluxT& transverse_flux,
//                                    std::size_t f_lo,
//                                    std::size_t f_hi,
//                                    double coeff,
//                                    double gamma) noexcept {
//     prims q0p = load_face_prims(out_faces, out_idx);
//     cons  q0  = aether::phys::prims_to_cons_cell(q0p, gamma);
//     cons  qc  = add_flux_diff(q0, transverse_flux, f_lo, f_hi, coeff);
//     prims qp  = recover_prims(qc, gamma);
//     store_face_prims(out_faces, out_idx, qp);
// }

// template<class FaceOutT, class BackupT, class Flux1T, class Flux2T>
// AETHER_INLINE void write_half_step_with_fallback(FaceOutT& out_faces,
//                                                  std::size_t out_idx,
//                                                  BackupT& backup,
//                                                  Flux1T& F1,
//                                                  std::size_t f1_lo,
//                                                  std::size_t f1_hi,
//                                                  double c1,
//                                                  Flux2T& F2,
//                                                  std::size_t f2_lo,
//                                                  std::size_t f2_hi,
//                                                  double c2,
//                                                  double gamma) noexcept {
//     cons q0 = load_backup_cons(backup, out_idx);
//     cons qc = q0;

//     qc = add_flux_diff(qc, F1, f1_lo, f1_hi, c1);
//     qc = add_flux_diff(qc, F2, f2_lo, f2_hi, c2);

//     prims qp = recover_prims_or_fallback(qc, q0, gamma);
//     store_face_prims(out_faces, out_idx, qp);
// }

// } // namespace detail


// // ============================================================
// // 2D CTU half-step correction
// // ============================================================

// #if AETHER_DIM > 1
// template <>
// void ctu_half_time_correction<2>(Simulation& sim, Simulation::View& view) {
//     const int nx = sim.grid.nx;
//     const int ny = sim.grid.ny;
//     constexpr int k = 0;

//     const double gamma    = sim.grid.gamma;
//     const double dxt_half = 0.5 * sim.time.dt / sim.grid.dx;
//     const double dyt_half = 0.5 * sim.time.dt / sim.grid.dy;

//     auto& Lx = view.x_flux_left;
//     auto& Rx = view.x_flux_right;
//     auto& Fx = view.x_flux;

//     auto& Ly = view.y_flux_left;
//     auto& Ry = view.y_flux_right;
//     auto& Fy = view.y_flux;

//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int j = -1; j < ny + 1; ++j) {
//         for (int i = -1; i < nx + 1; ++i) {
//             const std::size_t x_right  = sim.flux_x_ext.index(i + 1, j, k);
//             const std::size_t x_left   = sim.flux_x_ext.index(i,     j, k);
//             const std::size_t y_top    = sim.flux_y_ext.index(i, j + 1, k);
//             const std::size_t y_bottom = sim.flux_y_ext.index(i, j,     k);

//             detail::correct_face_2d(Lx, x_right, Fy, y_bottom, y_top, -dyt_half, gamma);
//             detail::correct_face_2d(Rx, x_left,  Fy, y_bottom, y_top, -dyt_half, gamma);
//             detail::correct_face_2d(Ly, y_top,   Fx, x_left,   x_right, -dxt_half, gamma);
//             detail::correct_face_2d(Ry, y_bottom,Fx, x_left,   x_right, -dxt_half, gamma);
//         }
//     }
// }
// #endif


// // ============================================================
// // 3D CTU half-step correction
// // ============================================================

// #if AETHER_DIM > 2
// template <>
// void ctu_half_time_correction<3>(Simulation& sim, Simulation::View& view) {
//     const int nx = sim.grid.nx;
//     const int ny = sim.grid.ny;
//     const int nz = sim.grid.nz;

//     const double gamma     = sim.grid.gamma;
//     const double dxt_third = sim.time.dt / (3.0 * sim.grid.dx);
//     const double dyt_third = sim.time.dt / (3.0 * sim.grid.dy);
//     const double dzt_third = sim.time.dt / (3.0 * sim.grid.dz);

//     const double dxt_half  = 0.5 * sim.time.dt / sim.grid.dx;
//     const double dyt_half  = 0.5 * sim.time.dt / sim.grid.dy;
//     const double dzt_half  = 0.5 * sim.time.dt / sim.grid.dz;

//     auto ctu_view = sim.ctu_buff.view();

//     auto& Lx = view.x_flux_left;
//     auto& Rx = view.x_flux_right;
//     auto& Fx = view.x_flux;

//     auto& Ly = view.y_flux_left;
//     auto& Ry = view.y_flux_right;
//     auto& Fy = view.y_flux;

//     auto& Lz = view.z_flux_left;
//     auto& Rz = view.z_flux_right;
//     auto& Fz = view.z_flux;

//     auto& ctu_Lx = ctu_view.x_flux_left_view;
//     auto& ctu_Rx = ctu_view.x_flux_right_view;
//     auto& ctu_Fx = ctu_view.x_flux_view;

//     auto& ctu_Ly = ctu_view.y_flux_left_view;
//     auto& ctu_Ry = ctu_view.y_flux_right_view;
//     auto& ctu_Fy = ctu_view.y_flux_view;

//     auto& ctu_Lz = ctu_view.z_flux_left_view;
//     auto& ctu_Rz = ctu_view.z_flux_right_view;
//     auto& ctu_Fz = ctu_view.z_flux_view;

//     auto& xL_bak = ctu_view.x_left_backup;
//     auto& xR_bak = ctu_view.x_right_backup;
//     auto& yL_bak = ctu_view.y_left_backup;
//     auto& yR_bak = ctu_view.y_right_backup;
//     auto& zL_bak = ctu_view.z_left_backup;
//     auto& zR_bak = ctu_view.z_right_backup;

//     // --------------------------------------------------------
//     // first predictor stage
//     // --------------------------------------------------------
//     #pragma omp parallel for collapse(3) schedule(static)
//     for (int k = -1; k < nz + 1; ++k) {
//         for (int j = -1; j < ny + 1; ++j) {
//             for (int i = -1; i < nx + 1; ++i) {
//                 const std::size_t x_right  = sim.flux_x_ext.index(i + 1, j, k);
//                 const std::size_t x_left   = sim.flux_x_ext.index(i,     j, k);
//                 const std::size_t y_top    = sim.flux_y_ext.index(i, j + 1, k);
//                 const std::size_t y_bottom = sim.flux_y_ext.index(i, j,     k);
//                 const std::size_t z_up     = sim.flux_z_ext.index(i, j, k + 1);
//                 const std::size_t z_down   = sim.flux_z_ext.index(i, j, k);

//                 {
//                     prims q0p = detail::load_face_prims(Lx, x_right);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(xL_bak, x_right, q0);

//                     detail::store_face_prims(
//                         Lx, x_right,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fy, y_bottom, y_top, +dyt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Lx, x_right,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fz, z_down, z_up, +dzt_third), gamma));
//                 }

//                 {
//                     prims q0p = detail::load_face_prims(Rx, x_left);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(xR_bak, x_left, q0);

//                     detail::store_face_prims(
//                         Rx, x_left,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fy, y_bottom, y_top, +dyt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Rx, x_left,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fz, z_down, z_up, +dzt_third), gamma));
//                 }

//                 {
//                     prims q0p = detail::load_face_prims(Ly, y_top);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(yL_bak, y_top, q0);

//                     detail::store_face_prims(
//                         Ly, y_top,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fx, x_right, x_left, +dxt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Ly, y_top,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fz, z_down, z_up, +dzt_third), gamma));
//                 }

//                 {
//                     prims q0p = detail::load_face_prims(Ry, y_bottom);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(yR_bak, y_bottom, q0);

//                     detail::store_face_prims(
//                         Ry, y_bottom,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fx, x_right, x_left, +dxt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Ry, y_bottom,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fz, z_down, z_up, +dzt_third), gamma));
//                 }

//                 {
//                     prims q0p = detail::load_face_prims(Lz, z_up);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(zL_bak, z_up, q0);

//                     detail::store_face_prims(
//                         Lz, z_up,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fx, x_right, x_left, +dxt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Lz, z_up,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fy, y_bottom, y_top, +dyt_third), gamma));
//                 }

//                 {
//                     prims q0p = detail::load_face_prims(Rz, z_down);
//                     cons q0   = aether::phys::prims_to_cons_cell(q0p, gamma);
//                     detail::backup_cons(zR_bak, z_down, q0);

//                     detail::store_face_prims(
//                         Rz, z_down,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fx, x_right, x_left, +dxt_third), gamma));

//                     detail::store_face_prims(
//                         ctu_Rz, z_down,
//                         detail::recover_prims(
//                             detail::add_flux_diff(q0, Fy, y_bottom, y_top, +dyt_third), gamma));
//                 }
//             }
//         }
//     }

//     Riemann_dispatch(sim, view);
//     Riemann_dispatch(sim, ctu_view);

//     // --------------------------------------------------------
//     // half-step predictor stage
//     // --------------------------------------------------------
//     #pragma omp parallel for collapse(3) schedule(static)
//     for (int k = -1; k < nz + 1; ++k) {
//         for (int j = -1; j < ny + 1; ++j) {
//             for (int i = -1; i < nx + 1; ++i) {
//                 const std::size_t x_right  = sim.flux_x_ext.index(i + 1, j, k);
//                 const std::size_t x_left   = sim.flux_x_ext.index(i,     j, k);
//                 const std::size_t y_top    = sim.flux_y_ext.index(i, j + 1, k);
//                 const std::size_t y_bottom = sim.flux_y_ext.index(i, j,     k);
//                 const std::size_t z_up     = sim.flux_z_ext.index(i, j, k + 1);
//                 const std::size_t z_down   = sim.flux_z_ext.index(i, j, k);

//                 detail::write_half_step_with_fallback(
//                     Lx, x_right, xL_bak,
//                     ctu_Fy, y_top, y_bottom, +dyt_half,
//                     ctu_Fz, z_up,  z_down,   +dzt_half,
//                     gamma);

//                 detail::write_half_step_with_fallback(
//                     Rx, x_left, xR_bak,
//                     ctu_Fy, y_top, y_bottom, +dyt_half,
//                     ctu_Fz, z_up,  z_down,   +dzt_half,
//                     gamma);

//                 detail::write_half_step_with_fallback(
//                     Ly, y_top, yL_bak,
//                     ctu_Fx, x_right, x_left, +dxt_half,
//                     Fz,     z_up,    z_down, +dzt_half,
//                     gamma);

//                 detail::write_half_step_with_fallback(
//                     Ry, y_bottom, yR_bak,
//                     ctu_Fx, x_right, x_left, +dxt_half,
//                     Fz,     z_up,    z_down, +dzt_half,
//                     gamma);

//                 detail::write_half_step_with_fallback(
//                     Lz, z_up, zL_bak,
//                     Fx, x_right, x_left, +dxt_half,
//                     Fy, y_top,   y_bottom, +dyt_half,
//                     gamma);

//                 detail::write_half_step_with_fallback(
//                     Rz, z_down, zR_bak,
//                     Fx, x_right, x_left, +dxt_half,
//                     Fy, y_top,   y_bottom, +dyt_half,
//                     gamma);
//             }
//         }
//     }
// }
// #endif

// } // namespace aether::core