#pragma once
#include <cstddef>
#include <aether/core/config.hpp>
#include <cassert>

namespace aether::core {
    enum class Dir {X = 0, Y = 1, Z = 2}; // Useful container for direction and dimension, prolly doesn't belong here, will move later
    template <int DIM> struct ExtentsD;   // Templating on the number of dimentions of the problem

    template <>
    struct ExtentsD<1>{
        int nx = 0;                       // Interior domain sizes (No ghost cells)
        int Nx = 0;                       // Full domain sizes (Padded with ghost cells)
        std::size_t sx=1, sy=0, sz=0;     // Stide lengths for each axis (continuous in i)
        int ng = 0;                       // Number of ghost cells
        int Ny =1, Nz = 1;
        int ny = 0, nz = 0;
    
        //---------- Constructors ----------
        explicit ExtentsD() = default; 
        
        explicit ExtentsD(int nx_, int ng_): nx(nx_), ng(ng_){
            Nx = nx + 2*ng;
            Ny = 1, Nz = 1;
            sx = 1;
            sy = static_cast<std::size_t>(Nx);      // Stride immediately exceeds the domain. Useful for debugging
            sz = static_cast<std::size_t>(Ny) * Nx; // = Nx
        }
        
        //---------- Tollerant of 2/3D Constructors ----------
        explicit ExtentsD(int nx_,[[maybe_unused]] int, int ng_): ExtentsD(nx_,ng_){}
        explicit ExtentsD(int nx_,[[maybe_unused]] int, [[maybe_unused]]int, int ng_): ExtentsD(nx_,ng_){}


        //---------- Inline index function ----------
        AETHER_INLINE std::size_t index(int i) const{
        #if AETHER_BOUNDS_CHECK                     // Optional out of bounds access checking 
            assert(i >= -ng && i < nx + ng );
        #endif
            return static_cast<std::size_t>(i+ng);
        }
        AETHER_INLINE std::size_t index(int i, [[maybe_unused]] int j,[[maybe_unused]] int k) const{
        #if AETHER_BOUNDS_CHECK                     // Optional out of bounds access checking 
            assert(j == 0 && k == 0);
        #endif            
            return index(i);
        }
        AETHER_INLINE std::size_t flat() const {return static_cast<std::size_t>(Nx);}
    };
    
    template <>
    struct ExtentsD<2>{
        int nx = 0, ny = 0;             // Interior domain sizes (No ghost cells)
        int Nx = 0, Ny = 0, Nz = 0;     // Full domain sizes (Padded with ghost cells)
        std::size_t sx=1, sy=0, sz=0;   // Stide lengths for each axis (continuous in i)
        int ng = 0;                     // Number of ghost cells
        int nz = 0;
    
        //---------- Constructors ----------
        
        explicit ExtentsD() = default; 
        explicit ExtentsD(int nx_, int ny_, int ng_): nx(nx_), ny(ny_), ng(ng_){
            Nx = nx + 2*ng;
            Ny = ny + 2*ng;
            Nz = 1;
            sx = 1;
            nz = 0;
            sy = static_cast<std::size_t>(Nx);      // Stride in Y
            sz = static_cast<std::size_t>(Ny) * Nx; // Stride immediately exceeds the domain
        }
        explicit ExtentsD(int nx_, int ny_, [[maybe_unused]] int, int ng_): ExtentsD(nx_,ny_,ng_){}

        //---------- Inline index function ----------
        AETHER_INLINE std::size_t index(int i, int j) const{
        #if AETHER_BOUNDS_CHECK                     // Optional out of bounds access checking 
            assert(i >= -ng && i < nx + ng );
            assert(j >= -ng && j < ny + ng );
        #endif
            const int ii = i + ng; 
            const int jj = j + ng; 
            return static_cast<std::size_t>(ii) + static_cast<std::size_t>(jj)*sy;
        }

        AETHER_INLINE std::size_t index(int i, int j, [[maybe_unused]]int k) const{
        #if AETHER_BOUNDS_CHECK                     // Optional out of bounds access checking 
            assert(k == 0);
        #endif
        return index(i,j);
        }

        AETHER_INLINE std::size_t flat() const {return static_cast<std::size_t>(Nx)*Ny;}
    };

    template <>
    struct ExtentsD<3>{
        int nx = 0, ny = 0, nz = 0;         // Interior domain sizes (No ghost cells)
        int Nx = 0, Ny = 0, Nz = 0;         // Full domain sizes (Padded with ghost cells)
        std::size_t sx=1, sy=0, sz = 0;     // Stide lengths for each axis (continuous in i)
        int ng = 0;                         // Number of ghost cells
    
        //---------- Constructors ----------
        
        ExtentsD() = default; 
        ExtentsD(int nx_, int ny_, int nz_, int ng_): nx(nx_), ny(ny_), nz(nz_), ng(ng_){
            Nx = nx + 2*ng;
            Ny = ny + 2*ng;
            Nz = nz + 2*ng;

            sy = static_cast<std::size_t>(Nx);           // Stride length for the y-axis
            sz = static_cast<std::size_t>(Nx) * Ny;      // Stride length for the z-axis
        }

        //---------- Inline index function ----------
        AETHER_INLINE std::size_t index(int i, int j, int k) const{
        #if AETHER_BOUNDS_CHECK                     // Optional out of bounds access checking 
            assert(i >= -ng && i < nx + ng );
            assert(j >= -ng && j < ny + ng );
            assert(k >= -ng && k < nz + ng );
        #endif
            const int ii = i + ng; 
            const int jj = j + ng; 
            const int kk = k + ng; 

            return static_cast<std::size_t>(ii) + static_cast<std::size_t>(jj)*sy + static_cast<std::size_t>(kk)*sz;
        }

        AETHER_INLINE std::size_t flat() const {return static_cast<std::size_t>(Nx)*Ny*Nz;}
    };

    using Extents = ExtentsD<AETHER_DIM>;

    // returns a pointer difference value for strides in different dimentions
    AETHER_INLINE std::ptrdiff_t step(const ExtentsD<1> &e, Dir d){
        assert(d == Dir::X);
        return std::ptrdiff_t(e.sx);
    }

    AETHER_INLINE std::ptrdiff_t step(const ExtentsD<2> &e, Dir d){
        switch (d) {
            case (Dir::X): return std::ptrdiff_t(e.sx); break; 
            case (Dir::Y): return std::ptrdiff_t(e.sy); break;
            default: assert(false && "Invalid 2D dimention"); return 0;
        
        }
    }

    AETHER_INLINE std::ptrdiff_t step(const ExtentsD<3> &e, Dir d){
        return (d == Dir::X) ? std::ptrdiff_t(e.sx) :
               (d == Dir::Y) ? std::ptrdiff_t(e.sy) :
                               std::ptrdiff_t(e.sz);
    }
}