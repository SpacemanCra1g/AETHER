#include <aether/core/space_solvers/fog.hpp>
#include <aether/core/stencil_templates.hpp>
#include <aether/core/SpaceDispatch.hpp>
#include <stdexcept>


namespace aether::core{

    template<sweep_dir dir>
    AETHER_INLINE void FOG_sweep(Simulation& Sim) noexcept {
    constexpr int numvar = aether::phys_ct::numvar;

    auto view = Sim.view();
    const auto ext = view.prim.ext;

    const int nx = ext.nx;
    const int ny = ext.ny;
    const int nz = ext.nz;

    int i0 = 0,  i1 = nx;
    int j0 = 0,  j1 = ny;
    int k0 = 0,  k1 = nz;

    if constexpr (dir == sweep_dir::x) { i0 = -1; i1 = nx + 1; }
    if constexpr (dir == sweep_dir::y) { j0 = -1; j1 = ny + 1; }
    if constexpr (dir == sweep_dir::z) { k0 = -1; k1 = nz + 1; }

    double* AETHER_RESTRICT* prim_comp = view.prim.comp.data();

    std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(ext.sx);
    std::ptrdiff_t sy = static_cast<std::ptrdiff_t>(ext.sy);
    std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(ext.sz);

    std::size_t cell = 0;
    std::size_t faceR = 0;
    std::size_t faceL = 0;

    if constexpr (dir == sweep_dir::x) {
        auto& FL = view.x_flux_left;
        auto& FR = view.x_flux_right;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {
            cell = ext.index(i, j, k);
            faceR = Sim.flux_x_ext.index(i, j, k);
            faceL = Sim.flux_x_ext.index(i+1, j, k);

            CellAccessor<numvar> A{prim_comp, cell, sx, sy, sz};
            Stencil1D<0, numvar, sweep_dir::x> S{A};

            FOG_face_from_stencil<numvar, sweep_dir::x>(S, FL, FR, faceL,faceR);
        }
    return;
    }
// This needs refactoring for multi-D problems
#if AETHER_DIM > 1
    if constexpr (dir == sweep_dir::y) {
        auto& FL = view.y_flux_left;
        auto& FR = view.y_flux_right;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            cell  = ext.index(i, j, k);

            // Mirror x-logic:
            // j runs [-1, ny] because j0=-1, j1=ny+1
            // faceR at (j), faceL at (j+1)
            faceR = Sim.flux_y_ext.index(i, j,   k);
            faceL = Sim.flux_y_ext.index(i, j+1, k);

            CellAccessor<numvar> A{prim_comp, cell, sx, sy, sz};
            Stencil1D<0, numvar, sweep_dir::y> S{A};

            FOG_face_from_stencil<numvar, sweep_dir::y>(S, FL, FR, faceL, faceR);
        }
        return;
    }
#endif

#if AETHER_DIM > 2
    if constexpr (dir == sweep_dir::z) {
        auto& FL = view.z_flux_left;
        auto& FR = view.z_flux_right;

        #pragma omp for collapse(3) schedule(static) nowait
        for (int k = k0; k < k1; ++k)
        for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {

            cell  = ext.index(i, j, k);

            // Mirror x-logic:
            // k runs [-1, nz] because k0=-1, k1=nz+1
            // faceR at (k), faceL at (k+1)
            faceR = Sim.flux_z_ext.index(i, j, k);
            faceL = Sim.flux_z_ext.index(i, j, k+1);

            CellAccessor<numvar> A{prim_comp, cell, sx, sy, sz};
            Stencil1D<0, numvar, sweep_dir::z> S{A};

            FOG_face_from_stencil<numvar, sweep_dir::z>(S, FL, FR, faceL, faceR);
        }
        return;
    }
#endif

    }

//---------- Space solver dispatcher ----------

void Space_solve(Simulation& Sim) {
    switch (Sim.cfg.solve) {
        case solver::fog: {
            #pragma omp parallel default(none) shared(Sim)
            {
                    FOG_sweep<sweep_dir::x>(Sim);
                #if AETHER_DIM > 1
                    FOG_sweep<sweep_dir::y>(Sim);
                #endif
                #if AETHER_DIM > 2
                    FOG_sweep<sweep_dir::z>(Sim);
                #endif
            }
            break;
        }

        default:
            throw std::runtime_error("Space_solve: unknown space solver");
    }
}

} // namespace aether::core
