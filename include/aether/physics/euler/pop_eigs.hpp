#pragma once
#include <aether/core/config.hpp>
#include "aether/core/config_build.hpp"
#include <aether/math/mats.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <aether/core/views.hpp>
#include "aether/core/char_struct.hpp"

namespace aether::physics::euler{
    

    AETHER_INLINE void fill_eigenvectors(
        prims &p
      , aether::core::one_cell_spectral_container &chars
      , const double gamma)
    {
        if constexpr (AETHER_DIM == 1) {
            const double a2 = p.p * gamma / p.rho;
            const double inv_a2 = 1.0/a2;
            const double a = std::sqrt(a2);
            const double inv_rho = 1.0/p.rho;
            const double inv_a = 1.0/a;
            // x_prim right eigenvectors 
            (*chars.x_right)(0,0) = 1.0;
            (*chars.x_right)(0,1) = inv_a2;
            (*chars.x_right)(0,2) = inv_a2;            

            (*chars.x_right)(1,0) = 0.0;
            (*chars.x_right)(1,1) = -1.0*inv_a*inv_rho;
            (*chars.x_right)(1,2) = 1.0*inv_a*inv_rho; 

            (*chars.x_right)(2,0) = 0.0;
            (*chars.x_right)(2,1) = 1.0;
            (*chars.x_right)(2,2) = 1.0;

            // x_prim left eigenvectors 
            (*chars.x_left)(0,0) = 1.0;
            (*chars.x_left)(0,1) = 0.0;
            (*chars.x_left)(0,2) = -inv_a2;            

            (*chars.x_left)(1,0) = 0.0;
            (*chars.x_left)(1,1) = -.5*a*p.rho;
            (*chars.x_left)(1,2) = 0.5;

            (*chars.x_left)(2,0) = 0.0;
            (*chars.x_left)(2,1) = 0.5*a*p.rho;
            (*chars.x_left)(2,2) = 0.5;

            // x_prim eigenvector
            (*chars.x_eigs) = {p.vx, p.vx - a, p.vx+a,0.0,0.0};
            
            
        
        } else if constexpr (AETHER_DIM == 2) {
            const double a2 = p.p * gamma / p.rho;
            const double inv_a2 = 1.0/a2;
            const double a = std::sqrt(a2);
            const double inv_rho = 1.0/p.rho;
            const double inv_a = 1.0/a;

            // x_prim right eigenvectors 
            (*chars.x_right)(0,0) = 1.0;
            (*chars.x_right)(0,1) = 0.0;
            (*chars.x_right)(0,2) = inv_a2;
            (*chars.x_right)(0,3) = inv_a2;            

            (*chars.x_right)(1,0) = 0.0;
            (*chars.x_right)(1,1) = 0.0;
            (*chars.x_right)(1,2) = -1.0*inv_a*inv_rho;
            (*chars.x_right)(1,3) = 1.0*inv_a*inv_rho; 

            (*chars.x_right)(2,0) = 0.0;
            (*chars.x_right)(2,1) = 1.0;
            (*chars.x_right)(2,2) = 0.0;
            (*chars.x_right)(2,3) = 0.0;

            (*chars.x_right)(3,0) = 0.0;
            (*chars.x_right)(3,1) = 0.0;
            (*chars.x_right)(3,2) = 1.0;
            (*chars.x_right)(3,3) = 1.0;

            // x_prim left eigenvectors 
            (*chars.x_left)(0,0) = 1.0;
            (*chars.x_left)(0,1) = 0.0;
            (*chars.x_left)(0,2) = 0.0;
            (*chars.x_left)(0,3) = -inv_a2;            

            (*chars.x_left)(1,0) = 0.0;
            (*chars.x_left)(1,1) = 0.0;
            (*chars.x_left)(1,2) = 1.0;
            (*chars.x_left)(1,3) = 0.0;

            (*chars.x_left)(2,0) = 0.0;
            (*chars.x_left)(2,1) = -.5*a*p.rho;
            (*chars.x_left)(2,2) = 0.0;
            (*chars.x_left)(2,3) = 0.5;

            (*chars.x_left)(3,0) = 0.0;
            (*chars.x_left)(3,1) = 0.5*a*p.rho;
            (*chars.x_left)(3,2) = 0.0;
            (*chars.x_left)(3,3) = 0.5;

            // x_prim eigenvalues 
            (*chars.x_eigs) = {p.vx,p.vx, p.vx - a, p.vx+a,0.0};

            // y_prim right eigenvectors 
            (*chars.y_right)(0,0) = 1.0;
            (*chars.y_right)(0,1) = 0.0;
            (*chars.y_right)(0,2) = inv_a2;
            (*chars.y_right)(0,3) = inv_a2;            

            (*chars.y_right)(1,0) = 0.0;
            (*chars.y_right)(1,1) = 1.0;
            (*chars.y_right)(1,2) = 0.0;
            (*chars.y_right)(1,3) = 0.0;

            (*chars.y_right)(2,0) = 0.0;
            (*chars.y_right)(2,1) = 0.0;
            (*chars.y_right)(2,2) = -1.0*inv_a*inv_rho;
            (*chars.y_right)(2,3) = 1.0*inv_a*inv_rho; 

            (*chars.y_right)(3,0) = 0.0;
            (*chars.y_right)(3,1) = 0.0;
            (*chars.y_right)(3,2) = 1.0;
            (*chars.y_right)(3,3) = 1.0;

            // y_prim left eigenvectors 
            (*chars.y_left)(0,0) = 1.0;
            (*chars.y_left)(0,1) = 0.0;
            (*chars.y_left)(0,2) = 0.0;
            (*chars.y_left)(0,3) = -inv_a2;            

            (*chars.y_left)(1,0) = 0.0;
            (*chars.y_left)(1,1) = 1.0;
            (*chars.y_left)(1,2) = 0.0;
            (*chars.y_left)(1,3) = 0.0;

            (*chars.y_left)(2,0) = 0.0;
            (*chars.y_left)(2,1) = 0.0;
            (*chars.y_left)(2,2) = -.5*a*p.rho;
            (*chars.y_left)(2,3) = 0.5;

            (*chars.y_left)(3,0) = 0.0;
            (*chars.y_left)(3,1) = 0.0;
            (*chars.y_left)(3,2) = 0.5*a*p.rho;
            (*chars.y_left)(3,3) = 0.5;

            // y_prim eigenvalues 
            (*chars.y_eigs) = {p.vy,p.vy, p.vy - a, p.vy+a,0.0};            
        
        } else if constexpr (AETHER_DIM == 3) {
        
            const double a2 = p.p * gamma / p.rho;
            const double inv_a2 = 1.0/a2;
            const double a = std::sqrt(a2);
            const double inv_rho = 1.0/p.rho;
            const double inv_a = 1.0/a;

            // x_prim right eigenvectors 
            (*chars.x_right)(0,0) = 1.0;
            (*chars.x_right)(0,1) = 0.0;
            (*chars.x_right)(0,2) = 0.0;
            (*chars.x_right)(0,3) = inv_a2;
            (*chars.x_right)(0,4) = inv_a2;            

            (*chars.x_right)(1,0) = 0.0;
            (*chars.x_right)(1,1) = 0.0;
            (*chars.x_right)(1,2) = 0.0;
            (*chars.x_right)(1,3) = -1.0*inv_a*inv_rho;
            (*chars.x_right)(1,4) = 1.0*inv_a*inv_rho; 

            (*chars.x_right)(2,0) = 0.0;
            (*chars.x_right)(2,1) = 1.0;
            (*chars.x_right)(2,2) = 0.0;
            (*chars.x_right)(2,3) = 0.0;
            (*chars.x_right)(2,4) = 0.0;

            (*chars.x_right)(3,0) = 0.0;
            (*chars.x_right)(3,1) = 0.0;
            (*chars.x_right)(3,2) = 1.0;
            (*chars.x_right)(3,3) = 0.0;
            (*chars.x_right)(3,4) = 0.0;

            (*chars.x_right)(4,0) = 0.0;
            (*chars.x_right)(4,1) = 0.0;
            (*chars.x_right)(4,2) = 0.0;
            (*chars.x_right)(4,3) = 1.0;
            (*chars.x_right)(4,4) = 1.0;

            // x_prim left eigenvectors 
            (*chars.x_left)(0,0) = 1.0;
            (*chars.x_left)(0,1) = 0.0;
            (*chars.x_left)(0,2) = 0.0;
            (*chars.x_left)(0,3) = 0.0;
            (*chars.x_left)(0,4) = -inv_a2;            

            (*chars.x_left)(1,0) = 0.0;
            (*chars.x_left)(1,1) = 0.0;
            (*chars.x_left)(1,2) = 1.0;
            (*chars.x_left)(1,3) = 0.0;
            (*chars.x_left)(1,4) = 0.0;

            (*chars.x_left)(2,0) = 0.0;
            (*chars.x_left)(2,1) = 0.0;
            (*chars.x_left)(2,2) = 0.0;
            (*chars.x_left)(2,3) = 1.0;
            (*chars.x_left)(2,4) = 0.0;

            (*chars.x_left)(3,0) = 0.0;
            (*chars.x_left)(3,1) = -.5*a*p.rho;
            (*chars.x_left)(3,2) = 0.0;
            (*chars.x_left)(3,3) = 0.0;
            (*chars.x_left)(3,4) = 0.5;

            (*chars.x_left)(4,0) = 0.0;
            (*chars.x_left)(4,1) = 0.5*a*p.rho;
            (*chars.x_left)(4,2) = 0.0;
            (*chars.x_left)(4,3) = 0.0;
            (*chars.x_left)(4,4) = 0.5;

            // x_prim eigenvalues 
            (*chars.x_eigs) = {p.vx, p.vx, p.vx, p.vx - a, p.vx+a};

            // y_prim right eigenvectors 
            (*chars.y_right)(0,0) = 1.0;
            (*chars.y_right)(0,1) = 0.0;
            (*chars.y_right)(0,2) = 0.0;
            (*chars.y_right)(0,3) = inv_a2;
            (*chars.y_right)(0,4) = inv_a2;            

            (*chars.y_right)(1,0) = 0.0;
            (*chars.y_right)(1,1) = 1.0;
            (*chars.y_right)(1,2) = 0.0;
            (*chars.y_right)(1,3) = 0.0;
            (*chars.y_right)(1,4) = 0.0;

            (*chars.y_right)(2,0) = 0.0;
            (*chars.y_right)(2,1) = 0.0;
            (*chars.y_right)(2,2) = 0.0;
            (*chars.y_right)(2,3) = -1.0*inv_a*inv_rho;
            (*chars.y_right)(2,4) = 1.0*inv_a*inv_rho; 

            (*chars.y_right)(3,0) = 0.0;
            (*chars.y_right)(3,1) = 0.0;
            (*chars.y_right)(3,2) = 1.0;
            (*chars.y_right)(3,3) = 0.0;
            (*chars.y_right)(3,4) = 0.0;

            (*chars.y_right)(4,0) = 0.0;
            (*chars.y_right)(4,1) = 0.0;
            (*chars.y_right)(4,2) = 0.0;
            (*chars.y_right)(4,3) = 1.0;
            (*chars.y_right)(4,4) = 1.0;

            // y_prim left eigenvectors 
            (*chars.y_left)(0,0) = 1.0;
            (*chars.y_left)(0,1) = 0.0;
            (*chars.y_left)(0,2) = 0.0;
            (*chars.y_left)(0,3) = 0.0;
            (*chars.y_left)(0,4) = -inv_a2;            

            (*chars.y_left)(1,0) = 0.0;
            (*chars.y_left)(1,1) = 1.0;
            (*chars.y_left)(1,2) = 0.0;
            (*chars.y_left)(1,3) = 0.0;
            (*chars.y_left)(1,4) = 0.0;

            (*chars.y_left)(2,0) = 0.0;
            (*chars.y_left)(2,1) = 0.0;
            (*chars.y_left)(2,2) = 0.0;
            (*chars.y_left)(2,3) = 1.0;
            (*chars.y_left)(2,4) = 0.0;

            (*chars.y_left)(3,0) = 0.0;
            (*chars.y_left)(3,1) = 0.0;
            (*chars.y_left)(3,2) = -.5*a*p.rho;
            (*chars.y_left)(3,3) = 0.0;
            (*chars.y_left)(3,4) = 0.5;

            (*chars.y_left)(4,0) = 0.0;
            (*chars.y_left)(4,1) = 0.0;            
            (*chars.y_left)(4,2) = 0.5*a*p.rho;
            (*chars.y_left)(4,3) = 0.0;
            (*chars.y_left)(4,4) = 0.5;

            // y_prim eigenvalues 
            (*chars.y_eigs) = {p.vy, p.vy, p.vy, p.vy - a, p.vy+a};


            // z_prim right eigenvectors 
            (*chars.z_right)(0,0) = 1.0;
            (*chars.z_right)(0,1) = 0.0;
            (*chars.z_right)(0,2) = 0.0;
            (*chars.z_right)(0,3) = inv_a2;
            (*chars.z_right)(0,4) = inv_a2;            

            (*chars.z_right)(1,0) = 0.0;
            (*chars.z_right)(1,1) = 1.0;
            (*chars.z_right)(1,2) = 0.0;
            (*chars.z_right)(1,3) = 0.0;
            (*chars.z_right)(1,4) = 0.0;

            (*chars.z_right)(2,0) = 0.0;
            (*chars.z_right)(2,1) = 0.0;
            (*chars.z_right)(2,2) = 1.0;
            (*chars.z_right)(2,3) = 0.0;
            (*chars.z_right)(2,4) = 0.0;

            (*chars.z_right)(3,0) = 0.0;
            (*chars.z_right)(3,1) = 0.0;
            (*chars.z_right)(3,2) = 0.0;
            (*chars.z_right)(3,3) = -1.0*inv_a*inv_rho;
            (*chars.z_right)(3,4) = 1.0*inv_a*inv_rho; 

            (*chars.z_right)(4,0) = 0.0;
            (*chars.z_right)(4,1) = 0.0;
            (*chars.z_right)(4,2) = 0.0;
            (*chars.z_right)(4,3) = 1.0;
            (*chars.z_right)(4,4) = 1.0;

            // z_prim left eigenvectors 
            (*chars.z_left)(0,0) = 1.0;
            (*chars.z_left)(0,1) = 0.0;
            (*chars.z_left)(0,2) = 0.0;
            (*chars.z_left)(0,3) = 0.0;
            (*chars.z_left)(0,4) = -inv_a2;            

            (*chars.z_left)(1,0) = 0.0;
            (*chars.z_left)(1,1) = 1.0;
            (*chars.z_left)(1,2) = 0.0;
            (*chars.z_left)(1,3) = 0.0;
            (*chars.z_left)(1,4) = 0.0;

            (*chars.z_left)(2,0) = 0.0;
            (*chars.z_left)(2,1) = 0.0;
            (*chars.z_left)(2,2) = 1.0;
            (*chars.z_left)(2,3) = 0.0;
            (*chars.z_left)(2,4) = 0.0;

            (*chars.z_left)(3,0) = 0.0;
            (*chars.z_left)(3,1) = 0.0;
            (*chars.z_left)(3,2) = 0.0;
            (*chars.z_left)(3,3) = -.5*a*p.rho;
            (*chars.z_left)(3,4) = 0.5;

            (*chars.z_left)(4,0) = 0.0;
            (*chars.z_left)(4,1) = 0.0;            
            (*chars.z_left)(4,2) = 0.0;
            (*chars.z_left)(4,3) = 0.5*a*p.rho;
            (*chars.z_left)(4,4) = 0.5;

            // z_prim eigenvalues 
            (*chars.z_eigs) = {p.vz, p.vz, p.vz, p.vz - a, p.vz+a};

        } 
    }

    void calc_eigenvecs(aether::core::CellsView &prim_view,
                        aether::core::eigenvec_view &eigs,
                        const double gamma);
}