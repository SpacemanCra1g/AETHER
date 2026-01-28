#include <aether/core/simulation.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/api.hpp>

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

AETHER_INLINE void Riemann_dispatch(Simulation& Sim, double gamma) noexcept {
#pragma omp parallel default(none) shared(Sim,gamma)
{
    if (Sim.cfg.riem == riemann::hll){
    Riemann_sweep<riemann::hll,sweep_dir::x>(Sim, gamma);
    #if AETHER_DIM > 1
    Riemann_sweep<riemann::hll,sweep_dir::y>(Sim, gamma);
    #endif
    #if AETHER_DIM > 2
    Riemann_sweep<riemann::hll,sweep_dir::z>(Sim, gamma);
    #endif
    }
}
}

}; // namespace aether::core



/*
AETHER_INLINE static void HLL_dispatch(Simulation &Sim, const double gamma) noexcept{
    auto view = Sim.view();
    auto ext = view.prim.ext;
    const int nx = ext.nx;
    const int ny = ext.ny;
    const int nz = ext.nz;
    std::size_t face1, face2;

    #pragma omp parallel for schedule(static) collapse(3) default(none) \
    shared(nx,ny,nz,ext,view,Sim,gamma) private(face1,face2)
    for (int k = 0; k < nz; k++){
    for (int j = 0; j < ny; j++){
    for (int i = -1; i < nx; i++){
        face1 = Sim.flux_x_ext.index(i, j, k);
        face2 = Sim.flux_x_ext.index(i+1, j, k);
        aether::phys::prims L, R, F; 
        L.rho = view.x_flux_right.var(P::RHO,face1,1);
        L.vx = view.x_flux_right.var(P::VX,face1,1);
        L.vy = (AETHER_DIM > 1) ? view.x_flux_right.var(P::VY,face1,1) : 0.0;
        L.vz = (AETHER_DIM > 2) ? view.x_flux_right.var(P::VZ,face1,1) : 0.0;
        L.p = view.x_flux_right.var(P::P,face1,1);


        R.rho = view.x_flux_left.var(P::RHO,face2,1);
        R.vx = view.x_flux_left.var(P::VX,face2,1);
        R.vy = (AETHER_DIM > 1) ? view.x_flux_left.var(P::VY,face2,1) : 0.0;
        R.vz = (AETHER_DIM > 2) ? view.x_flux_left.var(P::VZ,face2,1) : 0.0;
        R.p = view.x_flux_left.var(P::P,face2,1);

        F = hll(L, R, gamma);

        view.x_flux_right.var(P::RHO,face1,1) = F.rho;
        view.x_flux_right.var(P::VX,face1,1) = F.vx;
        if constexpr (AETHER_DIM > 1){
            view.x_flux_right.var(P::VY,face1,1) = F.vy;
        } 
        if constexpr (AETHER_DIM > 2){
            view.x_flux_right.var(P::VZ,face1,1) = F.vz;
        } 
        view.x_flux_right.var(P::P,face1,1) = F.p;
    }}}

    #if AETHER_DIM > 1
    #pragma omp parallel for schedule(static) collapse(3) default(none) \
    shared(nx,ny,nz,ext,view,Sim,gamma) private(face1,face2)
    for (int k = 0; k < nz; k++){
    for (int j = -1; j < ny; j++){
    for (int i = 0; i < nx; i++){
        face1 = Sim.flux_y_ext.index(i, j, k);
        face2 = Sim.flux_y_ext.index(i, j+1, k);
        aether::phys::prims L, R, F; 
        L.rho = view.y_flux_right.var(P::RHO,face1,1);
        L.vx = view.y_flux_right.var(P::VY,face1,1);
        L.vy = view.y_flux_right.var(P::VX,face1,1);
        L.vz = (AETHER_DIM > 2) ? view.y_flux_right.var(P::VZ,face1,1) : 0.0;
        L.p = view.y_flux_right.var(P::P,face1,1);


        R.rho = view.y_flux_left.var(P::RHO,face2,1);
        R.vx = view.y_flux_left.var(P::VY,face2,1);
        R.vy = view.y_flux_left.var(P::VX,face2,1);
        R.vz = (AETHER_DIM > 2) ? view.y_flux_left.var(P::VZ,face2,1) : 0.0;        
        R.p = view.y_flux_left.var(P::P,face2,1);

        F = hll(L, R, gamma);

        view.y_flux_right.var(P::RHO,face1,1) = F.rho;
        view.y_flux_right.var(P::VX,face1,1) = F.vy;
        view.y_flux_right.var(P::VY,face1,1) = F.vx;
        if constexpr (AETHER_DIM > 2){
            view.y_flux_right.var(P::VZ,face1,1) = F.vz;
        } 
        view.y_flux_right.var(P::P,face1,1) = F.p;
    }}}
    #endif 

    #if AETHER_DIM > 2
    #pragma omp parallel for schedule(static) collapse(3) default(none) \
    shared(nx,ny,nz,ext,view,Sim,gamma) private(face1,face2)
    for (int k = -1; k < nz; k++){
    for (int j = 0; j < ny; j++){
    for (int i = 0; i < nx; i++){
        face1 = Sim.flux_z_ext.index(i, j, k);
        face2 = Sim.flux_z_ext.index(i, j, k+1);
        aether::phys::prims L, R, F; 
        L.rho = view.z_flux_right.var(P::RHO,face1,1);
        L.vx  = view.z_flux_right.var(P::VZ,face1,1);
        L.vy  = view.z_flux_right.var(P::VY,face1,1);
        L.vz  = view.z_flux_right.var(P::VX,face1,1);
        L.p   = view.z_flux_right.var(P::P,face1,1);


        R.rho = view.z_flux_left.var(P::RHO,face2,1);
        R.vx  = view.z_flux_left.var(P::VZ,face2,1);
        R.vy  = view.z_flux_left.var(P::VY,face2,1);
        R.vz  = view.z_flux_left.var(P::VX,face2,1);
        R.p   = view.z_flux_left.var(P::P,face2,1);

        F = hll(L, R, gamma);

        view.z_flux_right.var(P::RHO,face1,1) = F.rho;
        view.z_flux_right.var(P::VX,face1,1) = F.vz;
        view.z_flux_right.var(P::VY,face1,1) = F.vy;
        view.z_flux_right.var(P::VZ,face1,1) = F.vx;
        view.z_flux_right.var(P::P,face1,1) = F.p;
    }}}
    #endif
}

void Riemann(Simulation &Sim){
    switch (Sim.cfg.riem) {
        case riemann::hll : 
            HLL_dispatch(Sim, Sim.grid.gamma);  
            break;

        default:
            throw std::runtime_error("Unknown call in Riemann Dispatcher");
    }
}

}
*/