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
        auto& FR = view.x_flux_right;
        auto& FL = view.x_flux_left;
        auto& Flux = view.x_flux;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            std::size_t interface_idx = Sim.flux_x_ext.index(i+1, j, k);

            aether::phys::prims L{}, R{}, F{};

            L.rho = FL.comp[P::RHO][interface_idx];
            L.vx  = FL.comp[VelMap<dir>::VN][interface_idx];
            L.vy  = (P::HAS_VY) ? FL.comp[VelMap<dir>::VT1][interface_idx] : 0.0;
            L.vz  = (P::HAS_VZ) ? FL.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
            L.p   = FL.comp[P::P][interface_idx];

            R.rho = FR.comp[P::RHO][interface_idx];
            R.vx  = FR.comp[VelMap<dir>::VN][interface_idx];
            R.vy  = (P::HAS_VY) ? FR.comp[VelMap<dir>::VT1][interface_idx] : 0.0;
            R.vz  = (P::HAS_VZ) ? FR.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
            R.p   = FR.comp[P::P][interface_idx];


            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            Flux.comp[P::RHO][interface_idx] = F.rho;
            Flux.comp[VelMap<dir>::VN][interface_idx]  = F.vx;
            if constexpr (P::HAS_VY) Flux.comp[VelMap<dir>::VT1][interface_idx] = F.vy;
            if constexpr (P::HAS_VZ) Flux.comp[VelMap<dir>::VT2][interface_idx] = F.vz;
            Flux.comp[P::P][interface_idx]   = F.p;
        }
    }

#if AETHER_DIM > 1
    if constexpr (dir == sweep_dir::y) {
        auto& FR   = view.y_flux_right;
        auto& FL   = view.y_flux_left;
        auto& Flux = view.y_flux;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            // j0 = -1, j runs [-1, ny-1] => j+1 runs [0, ny]
            std::size_t interface_idx = Sim.flux_y_ext.index(i, j+1, k);

            aether::phys::prims L{}, R{}, F{};

            // Left state from FL
            L.rho = FL.comp[P::RHO][interface_idx];
            L.vx  = FL.comp[VelMap<dir>::VN][interface_idx];
            L.vy  = FL.comp[VelMap<dir>::VT1][interface_idx];
            L.vz  = (P::HAS_VZ) ? FL.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
            L.p   = FL.comp[P::P][interface_idx];

            // Right state from FR
            R.rho = FR.comp[P::RHO][interface_idx];
            R.vx  = FR.comp[VelMap<dir>::VN][interface_idx];
            R.vy  = FR.comp[VelMap<dir>::VT1][interface_idx];
            R.vz  = (P::HAS_VZ) ? FR.comp[VelMap<dir>::VT2][interface_idx] : 0.0;
            R.p   = FR.comp[P::P][interface_idx];

            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            Flux.comp[P::RHO][interface_idx] = F.rho;
            Flux.comp[VelMap<dir>::VN][interface_idx]  = F.vx;
            Flux.comp[VelMap<dir>::VT1][interface_idx] = F.vy;
            if constexpr (P::HAS_VZ) Flux.comp[VelMap<dir>::VT2][interface_idx] = F.vz;
            Flux.comp[P::P][interface_idx]   = F.p;
        }
    }
#endif

#if AETHER_DIM > 2
    if constexpr (dir == sweep_dir::z) {
        auto& FR   = view.z_flux_right;
        auto& FL   = view.z_flux_left;
        auto& Flux = view.z_flux;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            // k0 = -1, k runs [-1, nz-1] => k+1 runs [0, nz]
            std::size_t interface_idx = Sim.flux_z_ext.index(i, j, k+1);

            aether::phys::prims L{}, R{}, F{};

            // Left state from FL
            L.rho = FL.comp[P::RHO][interface_idx];
            L.vx  = FL.comp[VelMap<dir>::VN][interface_idx];
            L.vy  = FL.comp[VelMap<dir>::VT1][interface_idx];
            L.vz  = FL.comp[VelMap<dir>::VT2][interface_idx];
            L.p   = FL.comp[P::P][interface_idx];

            // Right state from FR
            R.rho = FR.comp[P::RHO][interface_idx];
            R.vx  = FR.comp[VelMap<dir>::VN][interface_idx];
            R.vy  = FR.comp[VelMap<dir>::VT1][interface_idx];
            R.vz  = FR.comp[VelMap<dir>::VT2][interface_idx];
            R.p   = FR.comp[P::P][interface_idx];

            if constexpr (solv == riemann::hll) {
                F = hll(L, R, gamma);
            }

            Flux.comp[P::RHO][interface_idx] = F.rho;
            Flux.comp[VelMap<dir>::VN][interface_idx]  = F.vx;
            if constexpr (P::) Flux.comp[VelMap<dir>::VT1][interface_idx] = F.vy;
            if constexpr (AETHER_DIM > 2) Flux.comp[VelMap<dir>::VT2][interface_idx] = F.vz;
            Flux.comp[P::P][interface_idx]   = F.p;
        }
    }
#endif

}

void Riemann_dispatch(Simulation& Sim, double gamma){
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
