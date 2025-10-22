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
template<int dim>static AETHER_INLINE void outflow_bc([[maybe_unused]]CellsViewT<dim> &var){
    throw std::runtime_error("Unknown BC Dims case");
};

template<>
void outflow_bc<1>(CellsViewT<1> &var){
    constexpr int numvar = phys_ct::numvar;
    const int ng = var.ext.ng;
    const int nx = var.ext.nx;

    for (int c = 0; c < numvar; ++c){
        double left_edge = var.var(c,0,0,0);
        double right_edge = var.var(c,nx-1,0,0);
        double* AETHER_RESTRICT p_left = var.comp[c];
        double* AETHER_RESTRICT p_right = &var.comp[c][nx];

        #pragma omp simd
        for (int i = 0; i < ng; ++i){
            p_left[i] = left_edge;
        }
        for (int i = 0; i < ng; ++i){
            p_right[i] = right_edge;
        }
    }
}

template<>
void outflow_bc<2>(CellsViewT<2> &var){
    constexpr int numvar = phys_ct::numvar;
    const int ng = var.ext.ng;
    const int nx = var.ext.nx;
    const int ny = var.ext.nx;

    for (int c = 0; c < numvar; ++c){
        
    }
}



// Dispatch function to call the boundary condition method
AETHER_INLINE void boundary_conditions(Simulation& Sim, CellsViewT<AETHER_DIM>& var){
    switch (Sim.cfg.bc) {
        case boundary_conditions::Outflow : outflow_bc<AETHER_DIM>(var); break;
        case boundary_conditions::Periodic : break;
        case boundary_conditions::Reflecting : break;
        default: throw std::runtime_error("Invalid Boundary Condition reached"); break;
    };
}
}