#pragma once
#include <cstddef>
#include <aether/core/config.hpp>
#include <cassert>

namespace aether::core {
    struct Extents{
        int nx = 0, ny = 0, nz = 0;         // Interior domain sizes (No ghost cells)
        int Nx = 0, Ny = 0, Nz = 0;         // Full domain sizes (Padded with ghost cells)
        std::size_t sx=1, sy=0, sz = 0;     // Stide lengths for each axis (continuous in i)
        int ng = 0;                         // Number of ghost cells
    
        //---------- Constructors ----------
        
        Extents() = default; 
        Extents(int nx_, int ny_, int nz_, int ng_): nx(nx_), ny(ny_), nz(nz_), ng(ng_){
            Nx = nx + 2*ng;
            Ny = ny + 2*ng;
            Nz = nz + 2*ng;

            sy = std::size_t(Nx);           // Stride length for the y-axis
            sz = std::size_t(Nx) * Ny;      // Stride length for the z-axis
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

            return std::size_t(ii) + std::size_t(jj)*sy + std::size_t(kk)*sz;
        }
    };

    enum class Dir {X = 0, Y = 1, Z = 2}; // Useful container for direction and dimention, prolly doesn't belong here, will move later

    // returns a pointer difference value for strides in different dimentions
    AETHER_INLINE std::ptrdiff_t step(const Extents &e, Dir d){
        return (d == Dir::X) ? std::ptrdiff_t(e.sx) :
               (d == Dir::Y) ? std::ptrdiff_t(e.sy) :
                               std::ptrdiff_t(e.sz);
    }

}