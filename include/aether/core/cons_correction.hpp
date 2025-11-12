#pragma once
#include <aether/core/views.hpp>
#include <aether/core/config.hpp>

namespace aether::core {

template <int numvar>
AETHER_INLINE void fourth_order_correction(CellsView &Cons, CellsView &Cons_point){
    const double coef1 = -1.0/24.0;
    const double coef2 = 13.0/12.0;
    const int nx = Cons.ext.nx;
    const int ng = Cons.ext.ng;

    if constexpr (AETHER_DIM == 1) {
        #pragma omp parallel for collapse(2) schedule(static) default(none) \
        shared(Cons,Cons_point,coef1,coef2,numvar,nx,ng)
        for (int var = 0; var < numvar; ++var){
            for (int i = 1-ng;i < nx+ng-1; i++){
                Cons_point.var(var,i,0,0) = 
                coef2*Cons.var(var,i,0,0) + 
                coef1*(Cons.var(var,i+1,0,0) + Cons.var(var,i-1,0,0));
            }
        }
        
    }else if constexpr (AETHER_DIM == 2) {
        const double coef3 = 7.0/6.0;
        const int ny = Cons.ext.ny;
        #pragma omp parallel for collapse(3) schedule(static) default(none) \
        shared(Cons,Cons_point,coef1,coef3,numvar,nx,ny,ng)
        for (int var = 0; var < numvar; ++var){
            for (int j = 1-ng;j < ny+ng-1; j++){
            for (int i = 1-ng;i < nx+ng-1; i++){
                Cons_point.var(var,i,j,0) = coef3*Cons.var(var,i,j,0) + 
                coef1*(Cons.var(var,i+1,j,0) + Cons.var(var,i-1,j,0) + 
                Cons.var(var,i,j+1,0) + Cons.var(var,i,j-1,0));
            }
            }
        }
    
    }else if constexpr (AETHER_DIM == 3) {
        const double coef4 = 5.0/4.0;
        const int ny = Cons.ext.ny;        
        const int nz = Cons.ext.nz;        

        #pragma omp parallel for collapse(4) schedule(static) default(none) \
        shared(Cons,Cons_point,coef1,coef4,numvar,nx,ny,nz,ng)
        for (int var = 0; var < numvar; ++var){
            for (int k = 1-ng;k < nz+ng-1; k++){
            for (int j = 1-ng;j < ny+ng-1; j++){
            for (int i = 1-ng;i < nx+ng-1; i++){
                Cons_point.var(var,i,j,k) = coef4*Cons.var(var,i,j,k) + 
                coef1*(Cons.var(var,i+1,j,k) + Cons.var(var,i-1,j,k) + 
                Cons.var(var,i,j+1,k) + Cons.var(var,i,j-1,k) + 
                Cons.var(var,i,j,k+1) + Cons.var(var,i,j,k-1));
            }
            }
            }
        }
    }
}
}