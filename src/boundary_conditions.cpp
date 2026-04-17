#include <Kokkos_Core.hpp>
#include <aether/core/config.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/counts.hpp>

#include <stdexcept>

namespace aether::core {

namespace {

[[maybe_unused]] KOKKOS_INLINE_FUNCTION
int clamp_index(const int idx, const int begin, const int end_exclusive) {
    return (idx < begin) ? begin : ((idx >= end_exclusive) ? (end_exclusive - 1) : idx);
}

KOKKOS_INLINE_FUNCTION
int wrap_index(const int idx, const int begin, const int end_exclusive) {
    const int n = end_exclusive - begin;
    int x = idx - begin;
    x %= n;
    if (x < 0) x += n;
    return begin + x;
}

// ============================================================
// Outflow BC
// ============================================================

template<typename Sim>
AETHER_INLINE void outflow_bc(Sim& sim, typename Sim::CellView var) {
    constexpr int numvar = phys_ct::numvar;
    using exec_space = typename Sim::policy_type::execution_space;

    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int ng = sim.grid.ng;

    if constexpr (Sim::dim == 1) {
        Kokkos::parallel_for(
            "bc_outflow_1d_xleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
                {0, 0}, {numvar, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int g) {
                var(c, 0, 0, g) = var(c, 0, 0, ib);
            }
        );

        Kokkos::parallel_for(
            "bc_outflow_1d_xright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
                {0, 0}, {numvar, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int g) {
                var(c, 0, 0, ie + g) = var(c, 0, 0, ie - 1);
            }
        );
    }

    else if constexpr (Sim::dim == 2) {
        const int jb = sim.cells.jbegin();
        const int je = sim.cells.jend();

        // X ghost slabs: all j, ghost i
        Kokkos::parallel_for(
            "Dim=2 Outflow BCs",
            Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<2>>(
                {0,0},{sim.cells.Ny,sim.cells.Nx}
            ),
            KOKKOS_LAMBDA(const int j, const int i){
                if (j < jb && i >= ib && i < ie) {
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,jb,i);
                } else if (j >= je && i >= ib && i < ie) {
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,je-1,i);
                } // This concludes the j sweep. Now for the i's                
                else if (i < ib && j >= jb && j < je) {
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,j,ib);
                } else if (i >= ie && j >= jb && j < je) {
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,j,ie-1);
                }
                // Now we sweep the corners 
                else if (i < ib && j < jb){
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,jb,ib);
                } else if (i < ib && j >= je){
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,je-1,ib);
                } else if (i >= ie && j >= je){
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,je-1,ie-1);
                } else if (i >= ie && j < jb){
                    for (int c = 0; c < numvar; ++c) var(c,0,j,i) = var(c,0,jb,ie-1);
                }

            }
        );
    }

    else if constexpr (Sim::dim == 3) {
        const int jb = sim.cells.jbegin();
        const int je = sim.cells.jend();
        const int kb = sim.cells.kbegin();
        const int ke = sim.cells.kend();

        // X ghost slabs: all j,k ; ghost i
        Kokkos::parallel_for(
            "bc_outflow_3d_xleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, 0}, {numvar, sim.cells.Nz, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int j, const int g) {
                const int sk = clamp_index(k, kb, ke);
                const int sj = clamp_index(j, jb, je);
                var(c, k, j, g) = var(c, sk, sj, ib);
            }
        );

        Kokkos::parallel_for(
            "bc_outflow_3d_xright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, 0}, {numvar, sim.cells.Nz, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int j, const int g) {
                const int sk = clamp_index(k, kb, ke);
                const int sj = clamp_index(j, jb, je);
                var(c, k, j, ie + g) = var(c, sk, sj, ie - 1);
            }
        );

        // Y ghost slabs: all k, interior i ; ghost j
        Kokkos::parallel_for(
            "bc_outflow_3d_yleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, ib}, {numvar, sim.cells.Nz, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int g, const int i) {
                const int sk = clamp_index(k, kb, ke);
                var(c, k, g, i) = var(c, sk, jb, i);
            }
        );

        Kokkos::parallel_for(
            "bc_outflow_3d_yright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, ib}, {numvar, sim.cells.Nz, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int g, const int i) {
                const int sk = clamp_index(k, kb, ke);
                var(c, k, je + g, i) = var(c, sk, je - 1, i);
            }
        );

        // Z ghost slabs: interior i,j ; ghost k
        Kokkos::parallel_for(
            "bc_outflow_3d_zleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, jb, ib}, {numvar, ng, je, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int j, const int i) {
                var(c, g, j, i) = var(c, kb, j, i);
            }
        );

        Kokkos::parallel_for(
            "bc_outflow_3d_zright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, jb, ib}, {numvar, ng, je, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int j, const int i) {
                var(c, ke + g, j, i) = var(c, ke - 1, j, i);
            }
        );
    }
}

// ============================================================
// Periodic BC
// ============================================================

template<typename Sim>
AETHER_INLINE void periodic_bc(Sim& sim, typename Sim::CellView var) {
    constexpr int numvar = phys_ct::numvar;
    using exec_space = typename Sim::policy_type::execution_space;

    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int ng = sim.grid.ng;

    if constexpr (Sim::dim == 1) {
        Kokkos::parallel_for(
            "bc_periodic_1d_xleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
                {0, 0}, {numvar, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int g) {
                var(c, 0, 0, g) = var(c, 0, 0, wrap_index(g, ib, ie));
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_1d_xright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
                {0, 0}, {numvar, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int g) {
                var(c, 0, 0, ie + g) = var(c, 0, 0, wrap_index(ie + g, ib, ie));
            }
        );
    }

    else if constexpr (Sim::dim == 2) {
        const int jb = sim.cells.jbegin();
        const int je = sim.cells.jend();

        // X ghost slabs: all j, ghost i
        Kokkos::parallel_for(
            "bc_periodic_2d_xleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
                {0, 0, 0}, {numvar, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int j, const int g) {
                const int sj = wrap_index(j, jb, je);
                const int si = wrap_index(g, ib, ie);
                var(c, 0, j, g) = var(c, 0, sj, si);
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_2d_xright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
                {0, 0, 0}, {numvar, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int j, const int g) {
                const int sj = wrap_index(j, jb, je);
                const int ii = ie + g;
                const int si = wrap_index(ii, ib, ie);
                var(c, 0, j, ii) = var(c, 0, sj, si);
            }
        );

        // Y ghost slabs: interior i only, ghost j
        Kokkos::parallel_for(
            "bc_periodic_2d_yleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
                {0, 0, ib}, {numvar, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int i) {
                const int sj = wrap_index(g, jb, je);
                var(c, 0, g, i) = var(c, 0, sj, i);
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_2d_yright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
                {0, 0, ib}, {numvar, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int i) {
                const int jj = je + g;
                const int sj = wrap_index(jj, jb, je);
                var(c, 0, jj, i) = var(c, 0, sj, i);
            }
        );
    }

    else if constexpr (Sim::dim == 3) {
        const int jb = sim.cells.jbegin();
        const int je = sim.cells.jend();
        const int kb = sim.cells.kbegin();
        const int ke = sim.cells.kend();

        // X ghost slabs: all j,k ; ghost i
        Kokkos::parallel_for(
            "bc_periodic_3d_xleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, 0}, {numvar, sim.cells.Nz, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int j, const int g) {
                const int sk = wrap_index(k, kb, ke);
                const int sj = wrap_index(j, jb, je);
                const int si = wrap_index(g, ib, ie);
                var(c, k, j, g) = var(c, sk, sj, si);
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_3d_xright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, 0}, {numvar, sim.cells.Nz, sim.cells.Ny, ng}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int j, const int g) {
                const int sk = wrap_index(k, kb, ke);
                const int sj = wrap_index(j, jb, je);
                const int ii = ie + g;
                const int si = wrap_index(ii, ib, ie);
                var(c, k, j, ii) = var(c, sk, sj, si);
            }
        );

        // Y ghost slabs: all k, interior i ; ghost j
        Kokkos::parallel_for(
            "bc_periodic_3d_yleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, ib}, {numvar, sim.cells.Nz, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int g, const int i) {
                const int sk = wrap_index(k, kb, ke);
                const int sj = wrap_index(g, jb, je);
                var(c, k, g, i) = var(c, sk, sj, i);
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_3d_yright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, 0, ib}, {numvar, sim.cells.Nz, ng, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int k, const int g, const int i) {
                const int sk = wrap_index(k, kb, ke);
                const int jj = je + g;
                const int sj = wrap_index(jj, jb, je);
                var(c, k, jj, i) = var(c, sk, sj, i);
            }
        );

        // Z ghost slabs: interior i,j ; ghost k
        Kokkos::parallel_for(
            "bc_periodic_3d_zleft",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, jb, ib}, {numvar, ng, je, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int j, const int i) {
                const int sk = wrap_index(g, kb, ke);
                var(c, g, j, i) = var(c, sk, j, i);
            }
        );

        Kokkos::parallel_for(
            "bc_periodic_3d_zright",
            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>>(
                {0, 0, jb, ib}, {numvar, ng, je, ie}
            ),
            KOKKOS_LAMBDA(const int c, const int g, const int j, const int i) {
                const int kk = ke + g;
                const int sk = wrap_index(kk, kb, ke);
                var(c, kk, j, i) = var(c, sk, j, i);
            }
        );
    }
}

template<typename Sim>
AETHER_INLINE void DoubleMachReflection(Sim& sim, typename Sim::CellView var) {
    using exec_space = typename Sim::policy_type::execution_space;
    using P = aether::prim::Prim;

    double Gamma = sim.grid.gamma;
    const int ib = sim.cells.ibegin();
    const int ie = sim.cells.iend();
    const int jb = sim.cells.jbegin();
    const int je = sim.cells.jend();

    const double dx   = sim.grid.dx;
    const double xbeg = sim.grid.x_min + 0.5*dx;
    const double t    = sim.time.t;
    const int ng = sim.grid.ng;

    const double x0  = 1.0 / 6.0;
    const double alpha = M_PI/3.0;
    const double inv_sin_alpha = 1.0/std::sin(alpha);
    const double inv_tan_alpha = 1.0/std::tan(alpha);

    // Standard Woodward-Colella DMR primitive states for gamma = 1.4
    const double rho_pre  = 1.4;
    const double vx_pre   = 0.0;
    const double vy_pre   = 0.0;
    const double p_pre    = 1.0;
    double v2 = vx_pre*vx_pre + vy_pre*vy_pre;

    const double mx_pre = vx_pre*rho_pre;
    const double my_pre = vy_pre*rho_pre;
    const double E_pre = p_pre / (Gamma-1.0) + 0.5*rho_pre*v2;

    const double rho_post = 8.0;
    const double vx_post  = 8.25 * std::sin(alpha);
    const double vy_post  = -8.25 * std::cos(alpha); 
    const double p_post   = 116.5;

    v2 = vx_post*vx_post + vy_post*vy_post;
    const double mx_post  = vx_post*rho_post;
    const double my_post  = vy_post*rho_post;
    const double E_post   = p_post / (Gamma-1.0) + 0.5*rho_post*v2;

    Kokkos::parallel_for(
        "Dim=2 DoubleMachReflection BCs",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {sim.cells.Ny, sim.cells.Nx}
        ),
        KOKKOS_LAMBDA(const int j, const int i) {
            double x = xbeg + (i - ng)*dx;

            // Left boundary (post shock)
            if (i < ib ){
                var(P::RHO,0,j,i) = rho_post;
                var(P::VX,0,j,i) = mx_post;
                var(P::VY,0,j,i) = my_post;
                var(P::P,0,j,i) = E_post;
            }

            // Bottom boundary (post shock & recflecting)
            else if (j < jb && i >= ib && i < ie){
                if (x <= x0){
                    var(P::RHO,0,j,i) = rho_post;
                    var(P::VX,0,j,i) = mx_post;
                    var(P::VY,0,j,i) = my_post;
                    var(P::P,0,j,i) = E_post;
                } else{
                    var(P::RHO,0,j,i) = var(P::RHO, 0, 2*ng - j -1, i);
                    var(P::VX,0,j,i)  = var(P::VX,  0, 2*ng - j -1, i);
                    var(P::VY,0,j,i)  = - var(P::VY,  0, 2*ng - j -1, i);
                    var(P::P,0,j,i)   = var(P::P,   0, 2*ng - j -1, i);
                }
            }

            // Top boundary (Time dependent moving boundary)
            else if (j >= je && i >= ib && i < ie){
                double xs = 10.0*t*inv_sin_alpha + x0 + inv_tan_alpha;
                if (x < xs){
                    var(P::RHO,0,j,i) = rho_post;
                    var(P::VX,0,j,i) = mx_post;
                    var(P::VY,0,j,i) = my_post;
                    var(P::P,0,j,i) = E_post;
                } else{
                    var(P::RHO,0,j,i) = rho_pre;
                    var(P::VX,0,j,i) = mx_pre;
                    var(P::VY,0,j,i) = my_pre;
                    var(P::P,0,j,i) = E_pre;
                }
            }

            // Right boundary (pre-shock state)
            else if (i >= ie){
                var(P::RHO,0,j,i) = rho_pre;
                var(P::VX,0,j,i) = mx_pre;
                var(P::VY,0,j,i) = my_pre;
                var(P::P,0,j,i) = E_pre;
            }
        }
    );
}

} // unnamed namespace


#include <aether/core/boundary_conditions.hpp> 

void boundary_conditions(Simulation& sim, CellView var) {
    switch (sim.cfg.bc) {
        case boundary_conditions::Outflow:
            outflow_bc(sim, var);
            break;

        case boundary_conditions::Periodic:
            periodic_bc(sim, var);
            break;

        case boundary_conditions::DoubleMachReflection:
            DoubleMachReflection(sim, var);
            break;

        case boundary_conditions::Reflecting:
            // TODO
            break;

        default:
            throw std::runtime_error("Invalid boundary condition reached");
    }
}

} // namespace aether::core