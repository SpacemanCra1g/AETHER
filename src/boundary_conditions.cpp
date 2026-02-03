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
            const double left_edge  = var.var(c,0,0,0);
            const double right_edge = var.var(c,nx-1,0,0);
            double* AETHER_RESTRICT p_left  = var.comp[c];
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

        #pragma omp parallel default(none) shared(ng,nx,ny,var,numvar)
        {
            // x-faces (left/right), for all interior y
            #pragma omp for schedule(static) collapse(3) nowait
            for (int j = 0; j < ny; ++j){
                for (int c = 0; c < numvar; ++c){
                    for (int i = 0; i < ng; ++i){
                        var.var(c, i-ng, j, 0) = var.var(c, 0    , j , 0);
                        var.var(c, nx+i, j, 0) = var.var(c, nx-1 , j , 0);
                    }
                }
            }

            // y-faces (bottom/top), for all interior x
            #pragma omp for schedule(static) collapse(3) nowait
            for (int j = 0; j < ng; ++j){
                for (int c = 0; c < numvar; ++c){
                    for (int i = 0; i < nx; ++i){
                        var.var(c, i, j-ng  , 0) = var.var(c, i , 0    , 0);
                        var.var(c, i, ny+j , 0)  = var.var(c, i , ny-1 , 0);
                    }
                }
            }

            // 2D corners
            #pragma omp for schedule(static) collapse(3) nowait
            for (int c = 0; c < numvar; ++c){
                for (int jg = 0; jg < ng; ++jg){
                    for (int ig = 0; ig < ng; ++ig){
                        var.var(c, ig-ng, jg-ng, 0)   = var.var(c, 0   , 0    , 0);
                        var.var(c, nx+ig, jg-ng, 0)   = var.var(c, nx-1, 0    , 0);
                        var.var(c, ig-ng, ny+jg, 0)   = var.var(c, 0   , ny-1 , 0);
                        var.var(c, nx+ig, ny+jg, 0)   = var.var(c, nx-1, ny-1 , 0);
                    }
                }
            }
        } // OpenMP parallel
    } // End (Dim == 2)

    else if constexpr( AETHER_DIM == 3){

        const int ng = var.ext.ng;
        const int nx = var.ext.nx;
        const int ny = var.ext.ny;
        const int nz = var.ext.nz;

        #pragma omp parallel default(none) shared(ng,nx,ny,nz,var,numvar)
        {
            // -------------------------
            // Faces (6)
            // -------------------------

            // x-faces: i in ghost, j,k interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int k = 0; k < nz; ++k){
                for (int j = 0; j < ny; ++j){
                    for (int c = 0; c < numvar; ++c){
                        for (int ig = 0; ig < ng; ++ig){
                            var.var(c, ig-ng, j, k) = var.var(c, 0    , j, k);
                            var.var(c, nx+ig, j, k) = var.var(c, nx-1 , j, k);
                        }
                    }
                }
            }

            // y-faces: j in ghost, i,k interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int k = 0; k < nz; ++k){
                for (int jg = 0; jg < ng; ++jg){
                    for (int c = 0; c < numvar; ++c){
                        for (int i = 0; i < nx; ++i){
                            var.var(c, i, jg-ng, k) = var.var(c, i, 0    , k);
                            var.var(c, i, ny+jg, k) = var.var(c, i, ny-1 , k);
                        }
                    }
                }
            }

            // z-faces: k in ghost, i,j interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int kg = 0; kg < ng; ++kg){
                for (int j = 0; j < ny; ++j){
                    for (int c = 0; c < numvar; ++c){
                        for (int i = 0; i < nx; ++i){
                            var.var(c, i, j, kg-ng) = var.var(c, i, j, 0    );
                            var.var(c, i, j, nz+kg) = var.var(c, i, j, nz-1 );
                        }
                    }
                }
            }

            // -------------------------
            // Edges (12)
            // Each edge has 2 ghost directions and 1 interior direction.
            // -------------------------

            // Edges parallel to x: (j ghost, k ghost), i interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int c = 0; c < numvar; ++c){
                for (int kg = 0; kg < ng; ++kg){
                    for (int jg = 0; jg < ng; ++jg){
                        for (int i = 0; i < nx; ++i){
                            // (y-, z-)
                            var.var(c, i, jg-ng , kg-ng ) = var.var(c, i, 0    , 0    );
                            // (y+, z-)
                            var.var(c, i, ny+jg , kg-ng ) = var.var(c, i, ny-1 , 0    );
                            // (y-, z+)
                            var.var(c, i, jg-ng , nz+kg ) = var.var(c, i, 0    , nz-1 );
                            // (y+, z+)
                            var.var(c, i, ny+jg , nz+kg ) = var.var(c, i, ny-1 , nz-1 );
                        }
                    }
                }
            }

            // Edges parallel to y: (i ghost, k ghost), j interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int c = 0; c < numvar; ++c){
                for (int kg = 0; kg < ng; ++kg){
                    for (int ig = 0; ig < ng; ++ig){
                        for (int j = 0; j < ny; ++j){
                            // (x-, z-)
                            var.var(c, ig-ng , j, kg-ng ) = var.var(c, 0    , j, 0    );
                            // (x+, z-)
                            var.var(c, nx+ig , j, kg-ng ) = var.var(c, nx-1 , j, 0    );
                            // (x-, z+)
                            var.var(c, ig-ng , j, nz+kg ) = var.var(c, 0    , j, nz-1 );
                            // (x+, z+)
                            var.var(c, nx+ig , j, nz+kg ) = var.var(c, nx-1 , j, nz-1 );
                        }
                    }
                }
            }

            // Edges parallel to z: (i ghost, j ghost), k interior
            #pragma omp for schedule(static) collapse(4) nowait
            for (int c = 0; c < numvar; ++c){
                for (int jg = 0; jg < ng; ++jg){
                    for (int ig = 0; ig < ng; ++ig){
                        for (int k = 0; k < nz; ++k){
                            // (x-, y-)
                            var.var(c, ig-ng , jg-ng , k) = var.var(c, 0    , 0    , k);
                            // (x+, y-)
                            var.var(c, nx+ig , jg-ng , k) = var.var(c, nx-1 , 0    , k);
                            // (x-, y+)
                            var.var(c, ig-ng , ny+jg , k) = var.var(c, 0    , ny-1 , k);
                            // (x+, y+)
                            var.var(c, nx+ig , ny+jg , k) = var.var(c, nx-1 , ny-1 , k);
                        }
                    }
                }
            }

            // -------------------------
            // Corners (8)
            // 3 ghost directions: (i ghost, j ghost, k ghost)
            // -------------------------
            #pragma omp for schedule(static) collapse(4) nowait
            for (int c = 0; c < numvar; ++c){
                for (int kg = 0; kg < ng; ++kg){
                    for (int jg = 0; jg < ng; ++jg){
                        for (int ig = 0; ig < ng; ++ig){
                            // (x-, y-, z-)
                            var.var(c, ig-ng , jg-ng , kg-ng ) = var.var(c, 0    , 0    , 0    );
                            // (x+, y-, z-)
                            var.var(c, nx+ig , jg-ng , kg-ng ) = var.var(c, nx-1 , 0    , 0    );
                            // (x-, y+, z-)
                            var.var(c, ig-ng , ny+jg , kg-ng ) = var.var(c, 0    , ny-1 , 0    );
                            // (x+, y+, z-)
                            var.var(c, nx+ig , ny+jg , kg-ng ) = var.var(c, nx-1 , ny-1 , 0    );

                            // (x-, y-, z+)
                            var.var(c, ig-ng , jg-ng , nz+kg ) = var.var(c, 0    , 0    , nz-1 );
                            // (x+, y-, z+)
                            var.var(c, nx+ig , jg-ng , nz+kg ) = var.var(c, nx-1 , 0    , nz-1 );
                            // (x-, y+, z+)
                            var.var(c, ig-ng , ny+jg , nz+kg ) = var.var(c, 0    , ny-1 , nz-1 );
                            // (x+, y+, z+)
                            var.var(c, nx+ig , ny+jg , nz+kg ) = var.var(c, nx-1 , ny-1 , nz-1 );
                        }
                    }
                }
            }
        } // OpenMP parallel
    } // End (Dim == 3)


} 

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