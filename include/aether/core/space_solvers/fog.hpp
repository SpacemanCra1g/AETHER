#pragma once 
#include "aether/core/simulation.hpp"
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/counts.hpp>
#include <cstddef>

namespace aether::core{
    template<sweep_dir dir>
    AETHER_INLINE void FOG(Simulation &Sim) noexcept;

    template<>
    AETHER_INLINE void FOG<sweep_dir::x>(Simulation &Sim) noexcept{
        auto View = Sim.view();
        auto prims = View.prim;
        auto Cell_ext = prims.ext;
        const int nx = Cell_ext.nx;
        const int ny = Cell_ext.ny;
        const int nz = Cell_ext.nz;
        std::size_t face, cell;
        const int numvar = aether::phys_ct::numvar;

        #pragma omp parallel for schedule(static) collapse(3) 
        for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
        for (int i = -1; i < nx + 1; i++){
            face = Sim.flux_x_ext.index(i, j, k);
            cell = Cell_ext.index(i,j,k);
            for (int var = 0; var < numvar; ++var){
                View.x_flux_left.var(var, face, 1) = prims.var(var,cell);
                View.x_flux_right.var(var, face, 1) = prims.var(var,cell);
            }
        }}}
    };

#if AETHER_DIM > 1
    template<>
    AETHER_INLINE void FOG<sweep_dir::y>(Simulation &Sim) noexcept{
        auto View = Sim.view();
        auto prims = View.prim;
        auto Cell_ext = prims.ext;
        const int nx = Cell_ext.nx;
        const int ny = Cell_ext.ny;
        const int nz = Cell_ext.nz;
        std::size_t face, cell;
        const int numvar = aether::phys_ct::numvar;

        #pragma omp parallel for schedule(static) collapse(3) 
        for (int k = 0; k < nz; k++){
        for (int j = -1; j < ny+1; j++){
        for (int i = 0; i < nx; i++){
            face = Sim.flux_x_ext.index(i, j, k);
            cell = Cell_ext.index(i,j,k);
            for (int var = 0; var < numvar; ++var){
                View.y_flux_left.var(var, face, 1) = prims.var(var,cell);
                View.y_flux_right.var(var, face, 1) = prims.var(var,cell);
            }
        }}}
    };
#endif 

#if AETHER_DIM > 2
    template<>
    AETHER_INLINE void FOG<sweep_dir::z>(Simulation &Sim) noexcept{
        auto View = Sim.view();
        auto prims = View.prim;
        auto Cell_ext = prims.ext;
        const int nx = Cell_ext.nx;
        const int ny = Cell_ext.ny;
        const int nz = Cell_ext.nz;
        std::size_t face, cell;
        const int numvar = aether::phys_ct::numvar;

        #pragma omp parallel for schedule(static) collapse(3) 
        for (int k = -1; k < nz+1; k++){
        for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            face = Sim.flux_x_ext.index(i, j, k);
            cell = Cell_ext.index(i,j,k);
            for (int var = 0; var < numvar; ++var){
                View.z_flux_left.var(var, face, 1) = prims.var(var,cell);
                View.z_flux_right.var(var, face, 1) = prims.var(var,cell);
            }
        }}}
    };
#endif 
}