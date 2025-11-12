#pragma once 
#include <stdexcept>
#include <vector>
#include "aether/core/RunParams.hpp"
#include "aether/core/simulation.hpp"
#include "aether/physics/counts.hpp"
#include <aether/core/views.hpp>
#include <aether/core/enums.hpp>


// Runtime based heap allocation for buffers needed by various RK substages
// Needs changing based on what will actually be used, but this template work fn
namespace aether::core {

struct substage_container{
    std::vector<CellsSoA> buffer;
    CellsView Cons_point;

    void init(Simulation &sim){
        auto ext = &sim.prims_container.ext;
        switch(sim.cfg.time_step){
            case time_stepper::char_trace:
                buffer.clear();
                buffer.reserve(1);
                buffer.emplace_back(CellsSoA(ext->nx,ext->ny,ext->nz,ext->ng));
                Cons_point.ext = buffer[0].ext;
                for (int c = 0; c < aether::phys_ct::numvar; ++c) {
                    Cons_point.comp[c] = buffer[0].comp[c].data();
                }
                break;
            default: 
            throw std::invalid_argument("Invalid pass to the Rk_buffer constructor");
            break;
        }
        
        

        
    }
};
}
