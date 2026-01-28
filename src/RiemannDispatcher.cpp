#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/api.hpp>
#include <stdexcept>

using P = aether::prim::Prim;
namespace aether::core{

template<sweep_dir dir>
struct VelMap;

template<> struct VelMap<sweep_dir::x> {
    static constexpr int VN  = P::VX;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VZ;
};

template<> struct VelMap<sweep_dir::y> {
    static constexpr int VN  = P::VY;
    static constexpr int VT1 = P::VX;
    static constexpr int VT2 = P::VZ;
};

template<> struct VelMap<sweep_dir::z> {
    static constexpr int VN  = P::VZ;
    static constexpr int VT1 = P::VY;
    static constexpr int VT2 = P::VX;
};

template<riemann solv, sweep_dir dir>
AETHER_INLINE void Riemann_sweep(Simulation& Sim, const double gamma) noexcept {
    auto view = Sim.view();
    auto ext  = view.prim.ext;
    const int nx = ext.nx, ny = ext.ny, nz = ext.nz;

    std::size_t face1, face2;

    // Loop bounds
    int i0 = 0, i1 = nx;
    int j0 = 0, j1 = ny;
    int k0 = 0, k1 = nz;

    if constexpr (dir == sweep_dir::x) { i0 = -1; }
    if constexpr (dir == sweep_dir::y) { j0 = -1; }
    if constexpr (dir == sweep_dir::z) { k0 = -1; }

    // Select the correct face-state views with compile-time branching
    // and compute face indices using the correct ext.
    if constexpr (dir == sweep_dir::x) {
        // Make sure to switch these two        
        auto& FL = view.x_flux_right;
        auto& FR = view.x_flux_left;

        #pragma omp for collapse(3) schedule(static) private(face1, face2) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            face1 = Sim.flux_x_ext.index(i, j, k);
            face2 = Sim.flux_x_ext.index(i + 1, j, k);

            const std::size_t idx1 = FR.flat(face1, 1);
            const std::size_t idx2 = FL.flat(face2, 1);

            aether::phys::prims L{}, R{}, F{};

            L.rho = FR.comp[P::RHO][idx1];
            L.vx  = FR.comp[VelMap<dir>::VN][idx1];
            L.vy  = (AETHER_DIM > 1) ? FR.comp[VelMap<dir>::VT1][idx1] : 0.0;
            L.vz  = (AETHER_DIM > 2) ? FR.comp[VelMap<dir>::VT2][idx1] : 0.0;
            L.p   = FR.comp[P::P][idx1];

            R.rho = FL.comp[P::RHO][idx2];
            R.vx  = FL.comp[VelMap<dir>::VN][idx2];
            R.vy  = (AETHER_DIM > 1) ? FL.comp[VelMap<dir>::VT1][idx2] : 0.0;
            R.vz  = (AETHER_DIM > 2) ? FL.comp[VelMap<dir>::VT2][idx2] : 0.0;
            R.p   = FL.comp[P::P][idx2];

            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            FR.comp[P::RHO][idx1] = F.rho;
            FR.comp[VelMap<dir>::VN][idx1]  = F.vx;
            if constexpr (AETHER_DIM > 1) FR.comp[VelMap<dir>::VT1][idx1] = F.vy;
            if constexpr (AETHER_DIM > 2) FR.comp[VelMap<dir>::VT2][idx1] = F.vz;
            FR.comp[P::P][idx1]   = F.p;
        }
    }

#if AETHER_DIM > 1
    if constexpr (dir == sweep_dir::y) {
        auto& FL = view.y_flux_right;
        auto& FR = view.y_flux_left;

        #pragma omp for collapse(3) schedule(static) private(face1, face2) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            face1 = Sim.flux_y_ext.index(i, j, k);
            face2 = Sim.flux_y_ext.index(i, j + 1, k);

            const std::size_t idx1 = FR.flat(face1, 1);
            const std::size_t idx2 = FL.flat(face2, 1);

            aether::phys::prims L{}, R{}, F{};

            L.rho = FR.comp[P::RHO][idx1];
            L.vx  = FR.comp[VelMap<dir>::VN][idx1];      // normal = VY, stored in solver slot vx
            L.vy  = FR.comp[VelMap<dir>::VT1][idx1];     // tangential = VX, stored in solver slot vy
            L.vz  = (AETHER_DIM > 2) ? FR.comp[VelMap<dir>::VT2][idx1] : 0.0;
            L.p   = FR.comp[P::P][idx1];

            R.rho = FL.comp[P::RHO][idx2];
            R.vx  = FL.comp[VelMap<dir>::VN][idx2];
            R.vy  = FL.comp[VelMap<dir>::VT1][idx2];
            R.vz  = (AETHER_DIM > 2) ? FL.comp[VelMap<dir>::VT2][idx2] : 0.0;
            R.p   = FL.comp[P::P][idx2];

            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            FR.comp[P::RHO][idx1] = F.rho;
            FR.comp[VelMap<dir>::VN][idx1]  = F.vx;      // write normal back to VY
            FR.comp[VelMap<dir>::VT1][idx1] = F.vy;      // write tangential back to VX
            if constexpr (AETHER_DIM > 2) FR.comp[VelMap<dir>::VT2][idx1] = F.vz;
            FR.comp[P::P][idx1]   = F.p;
        }
    }
#endif

#if AETHER_DIM > 2
    if constexpr (dir == sweep_dir::z) {
        auto& FL = view.z_flux_right;
        auto& FR = view.z_flux_left;

        #pragma omp for collapse(3) schedule(static) private(face1, face2) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            face1 = Sim.flux_z_ext.index(i, j, k);
            face2 = Sim.flux_z_ext.index(i, j, k + 1);

            const std::size_t idx1 = FR.flat(face1, 1);
            const std::size_t idx2 = FL.flat(face2, 1);

            aether::phys::prims L{}, R{}, F{};

            L.rho = FR.comp[P::RHO][idx1];
            L.vx  = FR.comp[VelMap<dir>::VN][idx1];      // normal = VZ -> solver vx
            L.vy  = FR.comp[VelMap<dir>::VT1][idx1];     // tangential = VY -> solver vy
            L.vz  = FR.comp[VelMap<dir>::VT2][idx1];     // tangential = VX -> solver vz
            L.p   = FR.comp[P::P][idx1];

            R.rho = FL.comp[P::RHO][idx2];
            R.vx  = FL.comp[VelMap<dir>::VN][idx2];
            R.vy  = FL.comp[VelMap<dir>::VT1][idx2];
            R.vz  = FL.comp[VelMap<dir>::VT2][idx2];
            R.p   = FL.comp[P::P][idx2];

            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            FR.comp[P::RHO][idx1] = F.rho;
            FR.comp[VelMap<dir>::VN][idx1]  = F.vx;      // back to VZ
            FR.comp[VelMap<dir>::VT1][idx1] = F.vy;      // back to VY
            FR.comp[VelMap<dir>::VT2][idx1] = F.vz;      // back to VX
            FR.comp[P::P][idx1]   = F.p;
        }
    }
#endif
}

AETHER_INLINE void Riemann_dispatch(Simulation& Sim, double gamma){
    switch (Sim.cfg.riem) {
        case riemann::hll:{
            #pragma omp parallel default(none) shared(Sim,gamma)
            {
                Riemann_sweep<riemann::hll,sweep_dir::x>(Sim, gamma);
                #if AETHER_DIM > 1
                Riemann_sweep<riemann::hll,sweep_dir::y>(Sim, gamma);
                #endif
                #if AETHER_DIM > 2
                Riemann_sweep<riemann::hll,sweep_dir::z>(Sim, gamma);
                #endif
            }
            break;
        }
        default:
            throw std::runtime_error("RiemannSolve: unknown space solver, Dispatch");
    }
}

}; // namespace aether::core
