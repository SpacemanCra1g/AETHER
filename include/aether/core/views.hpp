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
};

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
int NxF{0}, Ny{0}, Nz{0};
std::size_t sy{0}, sz{0};

FaceGridX() = default; 
explicit FaceGridX(const Extents &e){
    NxF = e.Nx + 1; Ny = e.Ny; Nz = e.Nz;
    sy = std::size_t(NxF); sz = std::size_t(NxF)*Ny;
}

AETHER_INLINE std::size_t index(int iF, int j, int k) const {
#if AETHER_BOUNDS_CHECK
    assert(iF >= 0 && iF < NxF);
    assert(j  >= 0 && j  < Ny);
    assert(k  >= 0 && k  < Nz);
#endif 

    return std::size_t(iF) + std::size_t(j)*sy + std::size_t(k)*sz;
}

AETHER_INLINE std::size_t nfaces() const{ return std::size_t(NxF)*Ny*Nz;}
};

struct FaceGridY{
int Nx{0}, NyF{0}, Nz{0};
std::size_t sy{0}, sz{0};

FaceGridY() = default; 
explicit FaceGridY(const Extents &e){
    Nx = e.Nx; NyF = e.Ny+1; Nz = e.Nz;
    sy = std::size_t(Nx); sz = std::size_t(Nx)*NyF;
}

AETHER_INLINE std::size_t index(int i, int jF, int k) const {
#if AETHER_BOUNDS_CHECK
    assert(i >= 0 && i < Nx);
    assert(jF  >= 0 && jF  < NyF);
    assert(k  >= 0 && k  < Nz);
#endif 

    return std::size_t(i) + std::size_t(jF)*sy + std::size_t(k)*sz;
}

AETHER_INLINE std::size_t nfaces() const{ return std::size_t(Nx)*NyF*Nz;}
};

struct FaceGridZ{
int Nx{0}, Ny{0}, NzF{0};
std::size_t sy{0}, sz{0};

FaceGridZ() = default; 
explicit FaceGridZ(const Extents &e){
    Nx = e.Nx; Ny = e.Ny; NzF = e.Nz+1;
    sy = std::size_t(Nx); sz = std::size_t(Nx)*Ny;
}

AETHER_INLINE std::size_t index(int i, int j, int kF) const {
#if AETHER_BOUNDS_CHECK
    assert(i >= 0 && i < Nx);
    assert(j  >= 0 && j  < Ny);
    assert(kF  >= 0 && kF  < NzF);
#endif 

    return std::size_t(i) + std::size_t(j)*sy + std::size_t(kF)*sz;
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

    return std::size_t(face_idx)*Q + q;
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
