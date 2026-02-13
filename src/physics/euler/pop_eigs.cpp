#include "aether/core/char_struct.hpp"
#include "aether/core/config_build.hpp"
#include <aether/core/prim_layout.hpp>
#include <aether/physics/euler/pop_eigs.hpp>
#include <cstddef>

using P = aether::prim::Prim;
namespace co = aether::core;

namespace aether::physics::euler{
    void calc_eigenvecs(co::CellsView &prim_view, co::eigenvec_view &eigs, const double gamma){
        const std::size_t N = prim_view.ext.flat();
        #pragma omp parallel for schedule(static) default(none) shared(prim_view,eigs,gamma,N)
        for (std::size_t i = 0; i < N; ++i){
            prims prim; 
            co::one_cell_spectral_container chars;

            prim.rho = prim_view.var(P::RHO,i);
            prim.vx = prim_view.var(P::VX,i);
            if constexpr (P::HAS_VY){
                 prim.vy = prim_view.var(P::VY,i);
            } else{prim.vy = 0.0; }

            if constexpr (P::HAS_VZ){
                 prim.vz = prim_view.var(P::VZ,i);
            } else{prim.vz = 0.0; }
            
            prim.p = prim_view.var(P::P,i);

            chars.x_left = &eigs.x_left[i];
            chars.x_right = &eigs.x_right[i];
            chars.x_eigs = &eigs.x_eigs[i];

            if constexpr (AETHER_DIM > 1) {
            chars.y_left  = &eigs.y_left[i] ;
            chars.y_right = &eigs.y_right[i];
            chars.y_eigs  = &eigs.y_eigs[i] ;
            } else {
            chars.y_left  =  nullptr;
            chars.y_right =  nullptr;
            chars.y_eigs  =  nullptr;
            }

            if constexpr (AETHER_DIM > 2) {
            chars.z_left  = &eigs.z_left[i] ;
            chars.z_right = &eigs.z_right[i];
            chars.z_eigs  = &eigs.z_eigs[i] ;
            } else {
            chars.z_left  =  nullptr;
            chars.z_right =  nullptr;
            chars.z_eigs  =  nullptr;
            }            
 
            fill_eigenvectors(prim, chars, gamma);
        }
        *eigs.populated = true;    
    }
}