#pragma once 
#include <array>
#include <vector>
#include <cstddef>
#include <cassert>
#include <aether/core/config.hpp>
#include <aether/core/strides.hpp>
#include <aether/physics/counts.hpp>


namespace aether::core {

// template<int DIM> struct IndexAdaptor; // Wrapper that converts index


// ---------- Start with a generic Cell View structure ----------
// ---------- Template on the number of components (compile time known, physics & dim) ----------
template<int NCOMP> 
struct CellsViewT{
    std::array<double* AETHER_RESTRICT, NCOMP> comp{};  // comp[NCOMP][Flat_index]
    Extents ext;                        // Extents struct defined in strides.hpp

    AETHER_INLINE std::size_t idx(int i, int j=0, int k=0) const {return ext.index(i,j,k);} // Contains the boundary checking built into Extents 

    // Access with fast flat accessor 
    AETHER_INLINE double & var(int c, std::size_t flat) {return comp[c][flat];}
    // ---------- Access overload for const instantiations of the Cell View ----------    
    AETHER_INLINE const double & var(int c, std::size_t flat) const {return comp[c][flat];}

    // Access with index function
    AETHER_INLINE double & var(int c, int i, int j = 0, int k = 0) {return comp[c][idx(i,j,k)];}
    // ---------- Access overload for const instantiations of the Cell View ----------    
    AETHER_INLINE const double & var(int c, int i, int j = 0, int k = 0) const {return comp[c][idx(i,j,k)];}

    // CellsViewT() = default; 
    // CellsViewT(const CellsViewT&) = default; 
    // CellsViewT(CellsViewT&&) = default; 
    // CellsViewT& operator=(const CellsViewT&) = default;
    // CellsViewT& operator=(CellsViewT&&) = default;
    // ~CellsViewT() = default;
};



// ---------- Continue with a generic Cell container structure ----------
template<int NCOMP>
struct CellsSoAT {
    std::array<std::vector<double>,NCOMP> comp; 
    Extents ext; 


    CellsSoAT() = default; // Default empty constructor 

    // ---------- Constructor creates Extents struct and ----------
    // ---------- Allocates the flat array comp[NumberOfVariables][Total_Cells_in_T_Domain] ----------
    CellsSoAT(int nx, int ny, int nz, int ng) : ext(nx,ny,nz,ng){
        const std::size_t N = size_t(ext.Nx) * ext.Ny * ext.Nz; 
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);
    }

    // ---------- Returns the total flattened size of each variable array ----------
    [[nodiscard]] AETHER_INLINE std::size_t size_flat() const{
        return std::size_t(ext.Nx) * ext.Ny * ext.Nz;
    }

    // ---------- Returns the CellsView template, used for cell access ----------
    [[nodiscard]] AETHER_INLINE struct CellsViewT<NCOMP> view() noexcept{
        struct CellsViewT<NCOMP> v; 
        v.ext = ext; 
        for (int c = 0; c < NCOMP; ++c) v.comp[c] = comp[c].data();
        return v;
    }

    // ---------- This is a set of operator overloads for the Cells SOA ----------
    // ---------- very useful for the RK timestepping ----------

    // ---------- assignment operator ----------           
    AETHER_INLINE CellsSoAT& operator=(const CellsSoAT& rhs) {
        if (this == &rhs) return *this;

        ext = rhs.ext;
        const std::size_t N = rhs.size_flat();
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);

        #pragma omp parallel for schedule(static) default(none) shared(rhs, N) collapse(2)
        for (int c = 0; c < NCOMP; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                comp[c][i] = rhs.comp[c][i];
            }
        }
        return *this;
    }

    // ---------- scalar multiply ----------
    AETHER_INLINE CellsSoAT& operator*=(double s) noexcept {
        const std::size_t N = size_flat();

        #pragma omp parallel for schedule(static) default(none) shared(s, N) collapse(2)
        for (int c = 0; c < NCOMP; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                comp[c][i] *= s;
            }
        }
        return *this;
    }

    // ---------- Assignment addition ----------
    AETHER_INLINE CellsSoAT& operator+=(const CellsSoAT& rhs) noexcept {

        const std::size_t N = size_flat();

        #pragma omp parallel for schedule(static) default(none) shared(rhs, N) collapse(2)
        for (int c = 0; c < NCOMP; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                comp[c][i] += rhs.comp[c][i];
            }
        }
        return *this;
    }

    // ---------- Assignment subtraction ----------
    AETHER_INLINE CellsSoAT& operator-=(const CellsSoAT& rhs) noexcept {

        const std::size_t N = size_flat();

        #pragma omp parallel for schedule(static) default(none) shared(rhs, N) collapse(2)
        for (int c = 0; c < NCOMP; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                comp[c][i] -= rhs.comp[c][i];
            }
        }
        return *this;
    }
};

// ---------- Value-returning operators for Cells SOA ----------
template<int NCOMP>
AETHER_INLINE CellsSoAT<NCOMP> operator+(CellsSoAT<NCOMP> lhs, const CellsSoAT<NCOMP>& rhs) {
    lhs += rhs;
    return lhs;
}

template<int NCOMP>
AETHER_INLINE CellsSoAT<NCOMP> operator-(CellsSoAT<NCOMP> lhs, const CellsSoAT<NCOMP>& rhs) {
    lhs -= rhs;
    return lhs;
}

template<int NCOMP>
AETHER_INLINE CellsSoAT<NCOMP> operator*(CellsSoAT<NCOMP> lhs, double s) {
    lhs *= s;
    return lhs;
}

template<int NCOMP>
AETHER_INLINE CellsSoAT<NCOMP> operator*(double s, CellsSoAT<NCOMP> rhs) {
    rhs *= s;
    return rhs;
}

// ---------- BLAS style kernels for Cells SOA ----------
template<int NCOMP>
AETHER_INLINE void axpy(CellsSoAT<NCOMP>& y, const double a, const CellsSoAT<NCOMP>& x) noexcept {
    const std::size_t N = y.size_flat();
    #pragma omp parallel for schedule(static) default(none) shared(y,x,a,N) collapse(2)
    for (int c = 0; c < NCOMP; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
           y.comp[c][i] += a * x.comp[c][i];
        }
    }
}

template<int NCOMP>
AETHER_INLINE void axpby(CellsSoAT<NCOMP>& y, const double b, const CellsSoAT<NCOMP>& x, const double a) noexcept {
    const std::size_t N = y.size_flat();
    #pragma omp parallel for schedule(static) default(none) shared(y,x,a,b,N) collapse(2)
    for (int c = 0; c < NCOMP; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            y.comp[c][i] = a * x.comp[c][i] + b * y.comp[c][i];
        }
    }
}


template<int NCOMP> 
struct CharViewT{
    std::array<double* AETHER_RESTRICT, NCOMP> comp{};  // comp[NCOMP][Flat_index]
    Extents ext;                        // Extents struct defined in strides.hpp
    std::size_t N;

    AETHER_INLINE std::size_t idx(int dim, int i, int j=0, int k=0) const {return ext.index(i,j,k) + dim*N;} // Contains the boundary checking built into Extents 

    // Access with fast flat accessor 
    AETHER_INLINE double & var(int dim, int c, std::size_t flat) {return comp[c][flat + dim*N];}
    // ---------- Access overload for const instantiations of the Cell View ----------    
    AETHER_INLINE const double & var(int dim, int c, std::size_t flat) const {return comp[c][flat + dim*N];}

    // Access with index function
    AETHER_INLINE double & var(int dim, int c, int i, int j = 0, int k = 0) {return comp[c][idx(dim,i,j,k)];}
    // ---------- Access overload for const instantiations of the Cell View ----------    
    AETHER_INLINE const double & var(int dim, int c, int i, int j = 0, int k = 0) const {return comp[c][idx(dim,i,j,k)];}

};

// A storage object for charcteristic variables in each dimension
template<int NCOMP>
struct CharSoAT {
    std::array<std::vector<double>,NCOMP> comp; 
    Extents ext; 

    CharSoAT() = default; // Default empty constructor 

    // ---------- Constructor creates Extents struct and ----------
    // ---------- Allocates the flat array comp[NumberOfVariables * Dim]
    // ---------- This gives the number of characteristics needed for each dim
    CharSoAT(int nx, int ny, int nz, int ng) : ext(nx,ny,nz,ng){
        const std::size_t N = ext.flat()*AETHER_DIM; 
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);
    }

    // ---------- Returns the total flattened size of each variable array ----------
    [[nodiscard]] AETHER_INLINE std::size_t size_flat() const{
        return (std::size_t(ext.Nx) * ext.Ny * ext.Nz)*AETHER_DIM;
    }

    // ---------- Returns the CellsView template, used for cell access ----------
    [[nodiscard]] AETHER_INLINE struct CharViewT<NCOMP> view() noexcept{
        struct CharViewT<NCOMP> v; 
        v.ext = ext; 
        v.N = ext.flat();
        for (int c = 0; c < NCOMP; ++c) v.comp[c] = comp[c].data();
        return v;
    }
};

// -------------------------------------------------------------
// ----- Now we make similar containers for the flux faces -----
// ------------- Containers have the flat structure  -----------
// -------------------------------------------------------------
// ---------- For the X-Faces        (Nx+1)*Ny*Nz --------------
// ---------- For the Y-Faces        Nx*(Ny+1)*Nz --------------
// ---------- For the Z-Faces        Nx*Ny*(Nz+1) --------------
// -------------------------------------------------------------

struct FaceGridX{
int NxF{0}, Ny{0}, Nz{0}, ng{0};
std::size_t sy{0}, sz{0};

FaceGridX() = default; 
explicit FaceGridX(const Extents &e){
    NxF = e.Nx + 1; Ny = e.Ny; Nz = e.Nz; ng = e.ng;
    sy = std::size_t(NxF); sz = std::size_t(NxF)*Ny;
}

AETHER_INLINE std::size_t index(int iF, int j, int k) const {
#if AETHER_BOUNDS_CHECK
    assert(iF >= -ng && iF < NxF - ng);
    #if AETHER_DIM > 2
        assert(k  >= -ng && k  < Nz - ng);
        assert(j  >= -ng && j  < Ny - ng);
    #elif AETHER_DIM > 1
        assert(j  >= -ng && j  < Ny - ng);
    #endif 
#endif 

    int ii = iF + ng;
    int jj = j;
    int kk = k;
    #if AETHER_DIM > 2
        kk += ng;
        jj += ng;
    #elif AETHER_DIM > 1
        jj += ng;
    #endif 
    return std::size_t(ii) + std::size_t(jj)*sy + std::size_t(kk)*sz;
}

AETHER_INLINE std::size_t nfaces() const{ return std::size_t(NxF)*Ny*Nz;}
};

struct FaceGridY{
int Nx{0}, NyF{0}, Nz{0}, ng{0};
std::size_t sy{0}, sz{0};

FaceGridY() = default; 
explicit FaceGridY(const Extents &e){
    Nx = e.Nx; NyF = e.Ny+1; Nz = e.Nz;
    sy = std::size_t(Nx); sz = std::size_t(Nx)*NyF;
    ng = e.ng;
}

AETHER_INLINE std::size_t index(int i, int jF, int k) const {
#if AETHER_BOUNDS_CHECK
    assert(i >= -ng && i < Nx-ng);
    assert(jF  >= -ng && jF  < NyF-ng);
    #if AETHER_DIM > 2
        assert(k  >= -ng && k  < Nz-ng);
    #endif
#endif 

    int ii = i + ng;
    int jj = jF + ng;
    int kk = k;
    #if AETHER_DIM > 2
        kk += ng;
    #endif

    return std::size_t(ii) + std::size_t(jj)*sy + std::size_t(kk)*sz;
}

AETHER_INLINE std::size_t nfaces() const{ return std::size_t(Nx)*NyF*Nz;}
};

struct FaceGridZ{
int Nx{0}, Ny{0}, NzF{0}, ng{0};
std::size_t sy{0}, sz{0};

FaceGridZ() = default; 
explicit FaceGridZ(const Extents &e){
    Nx = e.Nx; Ny = e.Ny; NzF = e.Nz+1;
    sy = std::size_t(Nx); sz = std::size_t(Nx)*Ny;
    ng = e.ng;
}

AETHER_INLINE std::size_t index(int i, int j, int kF) const {
#if AETHER_BOUNDS_CHECK
    assert(i >= -ng && i < Nx-ng);
    assert(j  >= -ng && j  < Ny -ng);
    assert(kF  >= -ng && kF  < NzF - ng);
#endif 
    const int ii = i +ng;
    const int jj = j +ng;
    const int kk = kF +ng;
    return std::size_t(ii) + std::size_t(jj)*sy + std::size_t(kk)*sz;
}

AETHER_INLINE std::size_t nfaces() const{ return std::size_t(Nx)*Ny*NzF;}
};


// -------------------------------------------------------------
// ----- SoA Data containers for the flux faces + Quadrature ---
// ------------- Containers have the flat structure  -----------
// -------------------------------------------------------------
// ------------------ [comp][face * Q + q] ---------------------
// -------------------------------------------------------------


// ---------- First, define light quadrature struct  -----------
// ---------- Contains number of quadrature points per face  -----------

struct Quadrature {
int Q{1}; // Default to 1 quadrature point.
    Quadrature() = default;
    Quadrature(int Q_) : Q(Q_){};
};

// ---------- Template for Flux Face views ----------
template<int NCOMP>
struct FaceArrayViewT {
    std::array<double * AETHER_RESTRICT, NCOMP> comp;
    std::size_t nfaces{0};
    int Q{1};

    AETHER_INLINE std::size_t flat(int face_idx, int q) const {

#if AETHER_BOUNDS_CHECK        
    assert(std::size_t(face_idx) < nfaces && face_idx >= 0);
    assert(q >= 0 && q < Q);
#endif

        // Removed the Quadrature control because it was causing bugs 
        // be careful, this may be a problem later
    return std::size_t(face_idx)*Q;
    }

    AETHER_INLINE double& var(int c, int face_idx, int q) {return comp[c][flat(face_idx,q)];}

    // ---------- Access overload for const instantiations of the Flux View ----------
    AETHER_INLINE const double& var(int c, int face_idx, int q) const {return comp[c][flat(face_idx,q)];}
};


// ---------- Owner and Container of Flux's ----------
template <int NCOMP>
struct FaceArraySoAT{
    std::array<std::vector<double>, NCOMP> comp;
    std::size_t nfaces{0};
    int Q{1};

    // ---------- Constructors ----------
    FaceArraySoAT() = default; 

    // ---------- Overloads for each of the flux face types ----------
    explicit FaceArraySoAT(const FaceGridX& g, Quadrature quad) : nfaces(g.nfaces()), Q(quad.Q){
        const std::size_t N = nfaces*std::size_t(Q);
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);
    }

    explicit FaceArraySoAT(const FaceGridY& g, Quadrature quad) : nfaces(g.nfaces()), Q(quad.Q){
        const std::size_t N = nfaces*std::size_t(Q);
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);
    }

    explicit FaceArraySoAT(const FaceGridZ& g, Quadrature quad) : nfaces(g.nfaces()), Q(quad.Q){
        const std::size_t N = nfaces*std::size_t(Q);
        for (int c = 0; c < NCOMP; ++c) comp[c].resize(N);
    }

    [[nodiscard]] AETHER_INLINE FaceArrayViewT<NCOMP> view() noexcept{
        struct FaceArrayViewT<NCOMP> v; 
        v.Q = Q; v.nfaces = nfaces;
        for (int c = 0; c < NCOMP; ++c) v.comp[c] = comp[c].data();
        return v;
    }

};

// ---------- Quick and dirty overload to get face index, for any grid type ----------
AETHER_INLINE int face_index(FaceGridX& gx, int iF, int j, int k) {return int(gx.index(iF, j, k));}
AETHER_INLINE int face_index(FaceGridY& gy, int i, int jF, int k) {return int(gy.index(i, jF, k));}
AETHER_INLINE int face_index(FaceGridZ& gz, int i, int j, int kF) {return int(gz.index(i, j, kF));}


    using CellsView = CellsViewT<aether::phys_ct::numvar>;
    using CellsSoA = CellsSoAT<aether::phys_ct::numvar>;
    using CharView = CharViewT<aether::phys_ct::numvar>;
    using CharSoA = CharSoAT<aether::phys_ct::numvar>;
    using FaceArrayView = FaceArrayViewT<aether::phys_ct::numvar>;
    using FaceArraySoA = FaceArraySoAT<aether::phys_ct::numvar>;
}
