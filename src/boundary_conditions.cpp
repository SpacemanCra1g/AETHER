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

AETHER_INLINE static void outflow_bc(CellsView &var){
    constexpr int numvar = phys_ct::numvar;

    if constexpr( AETHER_DIM == 1){
        const int ng = var.ext.ng;
        const int nx = var.ext.nx;

        for (int c = 0; c < numvar; ++c){
            const double left_edge = var.var(c,0,0,0);
            const double right_edge = var.var(c,nx-1,0,0);
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

    else if constexpr( AETHER_DIM == 2){

        const int ng = var.ext.ng;
        const int nx = var.ext.nx;
        const int ny = var.ext.ny;
        
        #pragma omp parallel default(none) shared(numvar, ng,nx,ny,var)
        {

            #pragma omp for schedule(static) collapse(3) nowait
            for (int j = 0; j < ny; ++j){
                for (int c = 0; c < numvar; ++c){
                    for (int i = 0; i < ng; ++i){
                        var.var(c, i-ng, j, 0) = var.var(c, 0    ,j ,0);
                        var.var(c, nx+i, j, 0) = var.var(c, nx-1 ,j ,0);
                    }
                }
            }

            #pragma omp for schedule(static) collapse(3) nowait
            for (int j = 0; j < ng; ++j){
                for (int c = 0; c < numvar; ++c){
                    for (int i = 0; i < nx; ++i){
                        var.var(c, i, j-ng  , 0) = var.var(c, i ,0    ,0);
                        var.var(c, i, ny+j, 0) = var.var(c, i ,ny-1 ,0);
                    }
                }
            }
            
            #pragma omp for schedule(static) collapse(3) nowait
            for (int c = 0; c < numvar; ++c){
                for (int jg = 0; jg < ng; ++jg){
                    for (int ig = 0; ig < ng; ++ig){
                        var.var(c,ig-ng,jg-ng,0) = var.var(c,0,0,0);
                        var.var(c,nx+ig,jg-ng,0) = var.var(c,nx-1,0,0);
                        var.var(c,ig-ng,ny+jg,0) = var.var(c,0,ny-1,0);                        
                        var.var(c,nx+ig,ny+jg,0) = var.var(c,nx-1,ny-1,0);                        
                    }
                }
            }

        } //OpenMP parallel boundary
    } // End (if Dim == 2)
} // End Outflow bc

AETHER_INLINE static void periodic_bc(CellsView &var){
    constexpr int numvar = phys_ct::numvar;

    if constexpr( AETHER_DIM == 1){
        const int ng = var.ext.ng;
        const int nx = var.ext.nx;

        for (int c = 0; c < numvar; ++c){
            double* AETHER_RESTRICT right_edge = &var.var(c,nx-ng,0,0);
            double* AETHER_RESTRICT left_edge = &var.var(c,0,0,0);

            double* AETHER_RESTRICT p_left = var.comp[c];
            double* AETHER_RESTRICT p_right = &var.comp[c][nx+ng];

            #pragma omp simd
            for (int i = 0; i < ng; ++i){
                p_left[i] = *(right_edge+i);
            }
            #pragma omp simd
            for (int i = 0; i < ng; ++i){
                p_right[i] = *(left_edge+i);
            }
        }
    }

    else if constexpr( AETHER_DIM == 2){

        const int ng = var.ext.ng;
        const int nx = var.ext.nx;
        const int ny = var.ext.ny;

        #pragma omp parallel default(none) shared(ng,nx,ny,var,numvar)
        {   
            #pragma omp for collapse(3) schedule(static) nowait
            for (int c = 0; c < numvar; ++c){
                for (int j = 0; j < ny; ++j){
                    for (int i = 0; i < ng; ++i){
                        var.var(c, i-ng, j, 0) = var.var(c, nx-ng+i, j ,0);
                        var.var(c, nx+i, j, 0) = var.var(c, i , j ,0);
                    }
                }
            } // These boundary conditions are making my head hurt 

        
            #pragma omp for collapse(3) schedule(static) nowait        
            for (int c = 0; c < numvar; ++c){
                for (int j = 0; j < ng; ++j){
                    for (int i = 0; i < nx; ++i){
                        // Bottom Ghosts            Top interior cells
                        var.var(c, i, j-ng , 0) = var.var(c, i ,ny - ng + j,0);
                        // Top Ghosts               Bottom interior cells
                        var.var(c, i, ny+j, 0) = var.var(c, i ,j ,0);
                    }
                }
            }

            #pragma omp for collapse(3) schedule(static) nowait        
            for (int c = 0; c < numvar; ++c){
                for (int jg = 0; jg < ng; ++jg){
                    for (int ig = 0; ig < ng; ++ig){
                        // Bottom left Ghotop_left =sts    Top right interior
                        var.var(c,ig-ng,jg-ng,0) = var.var(c, nx-ng+ig, ny-ng+jg, 0);
                        var.var(c, nx+ig, jg-ng, 0) = var.var(c, ig, ny-ng+jg, 0);                        
                        var.var(c,ig-ng,ny+jg,0) = var.var(c,nx-ng + ig, jg, 0);
                        var.var(c,nx+ig,ny+jg,0) = var.var(c,ig,jg,0);
                    }
                }
            } // c loop for
        } // openMP parallel end
    } // End 2D case
}

// Dispatch function to call the boundary condition method
void boundary_conditions(Simulation& Sim,CellsView& var){
    switch (Sim.cfg.bc) {
        case boundary_conditions::Outflow : outflow_bc(var); break;
        case boundary_conditions::Periodic : periodic_bc(var); break;
        case boundary_conditions::Reflecting : break;
        default: throw std::runtime_error("Invalid Boundary Condition reached"); break;
    };
} //boundary_conditions
} // Namespace