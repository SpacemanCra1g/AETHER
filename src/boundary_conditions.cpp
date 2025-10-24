#include <stdexcept>
#include "aether/core/config.hpp"
#include "aether/core/simulation.hpp"
#include "aether/core/views.hpp"
#include "aether/physics/counts.hpp"
#include <aether/core/boundary_conditions.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/enums.hpp>
#include <sys/cdefs.h>

namespace aether::core { 

using CellsView = CellsViewT<aether::phys_ct::numvar>;

template<int dim>static AETHER_INLINE void outflow_bc([[maybe_unused]]CellsView &var){
    throw std::runtime_error("Unknown BC Dims case");
};

template<>
void outflow_bc<1>(CellsView &var){
    constexpr int numvar = phys_ct::numvar;
    const int ng = var.ext.ng;
    const int nx = var.ext.nx;

    for (int c = 0; c < numvar; ++c){
        double left_edge = var.var(c,0,0,0);
        double right_edge = var.var(c,nx-1,0,0);
        double* AETHER_RESTRICT p_left = var.comp[c];
        double* AETHER_RESTRICT p_right = &var.comp[c][nx+ng];

        #pragma omp simd
        for (int i = 0; i < ng; ++i){
            p_left[i] = left_edge;
        }
        #pragma omp simd
        for (int i = 0; i < ng; ++i){
            p_right[i] = right_edge;
        }
    }
}

template<>
void outflow_bc<2>(CellsView &var){
    constexpr int numvar = phys_ct::numvar;
    const int ng = var.ext.ng;
    const int nx = var.ext.nx;
    const int ny = var.ext.ny;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int j = 0; j < ny; ++j){
        for (int c = 0; c < numvar; ++c){
            const double left_edge  = var.var(c, 0    ,j ,0);
            const double right_edge = var.var(c, nx-1 ,j ,0);
            #pragma omp simd
            for (int i = 0; i < ng; ++i){
                var.var(c, i-ng, j, 0) = left_edge;
                var.var(c, nx+i, j, 0) = right_edge;
            }
        }
    }

    #pragma omp parallel 
    {
        for (int j = 0; j < ng; ++j){
            for (int c = 0; c < numvar; ++c){
                #pragma omp for simd schedule(static) nowait
                for (int i = 0; i < nx; ++i){
                    const double bottom_edge  = var.var(c, i ,0    ,0);
                    const double top_edge     = var.var(c, i ,ny-1 ,0);
            
                    var.var(c, i, j-ng  , 0) = bottom_edge;
                    var.var(c, i, ny+j, 0) = top_edge;
                }
            }
        }
    } // parallel barrier 

    for (int c = 0; c < numvar; ++c){
        const double bot_left = var.var(c,0,0,0);
        const double bot_right = var.var(c,nx-1,0,0);

        const double top_left = var.var(c,0,ny-1,0);
        const double top_right =  var.var(c,nx-1,ny-1,0);

        for (int jg = 0; jg < ng; ++jg){
            const int jb = jg-ng;
            const int jt = ny+jg;
            #pragma omp simd
            for (int ig = 0; ig < ng; ++ig){
                var.var(c,ig-ng,jb,0) = bot_left;
            }
            #pragma omp simd
            for (int ig = 0; ig < ng; ++ig){
                var.var(c,nx+ig,jb,0) = bot_right;
            }
            #pragma omp simd
            for (int ig = 0; ig < ng; ++ig){
                var.var(c,ig-ng,jt,0) = top_left;
            }
            #pragma omp simd
            for (int ig = 0; ig < ng; ++ig){
                var.var(c,nx+ig,jt,0) = top_right;
            }
        }
    }
}



// Dispatch function to call the boundary condition method
AETHER_INLINE void boundary_conditions(Simulation& Sim,CellsViewT<4>& var){
    switch (Sim.cfg.bc) {
        case boundary_conditions::Outflow : outflow_bc<AETHER_DIM>(var); break;
        case boundary_conditions::Periodic : break;
        case boundary_conditions::Reflecting : break;
        default: throw std::runtime_error("Invalid Boundary Condition reached"); break;
    };
}
}