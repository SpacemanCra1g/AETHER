#include "aether/core/prim_layout.hpp"
#include "aether/physics/euler/linear_correction/linear_cor.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/TemporalDispatch.hpp>
#include <aether/io/metadata.hpp>

int main() {

    using namespace aether::core;
    {
    // feenableexcept(FE_INVALID);        
        Kokkos::initialize();

        // Load run parameters
        Config cfg;
        Simulation sim;

        load_run_parameters(cfg);
        check_run_parameters(cfg);
        sim = Simulation(cfg);

        // Set up the data write out formats 
        aether::io::snapshot_request snap;
        if (cfg.write_text){
            snap.formats.push_back(aether::io::output_format::plain_txt);
        } if (cfg.write_binary){
            snap.formats.push_back(aether::io::output_format::binary);
        }
        snap.output_dir = cfg.output_dir;
        std::cout << cfg.output_dir;
        snap.prefix = cfg.prefix;
        snap.include_ghosts = true;
        write_run_metadata(sim, snap);

        //  load initial conditions and prepare for run 
        initialize_domain(sim);
        auto domain = sim.view();        

        // Write time = 0 
        aether::io::write_snapshot(sim, snap);

        // Populate conservative variables and perform boundary conditions
        aether::phys::prims_to_cons_domain(sim);
        boundary_conditions(sim,domain.cons);  
        
        auto p_copy = Kokkos::View<double****>("prim0"
            , domain.prim.extent(0)
            , domain.prim.extent(1)
            , domain.prim.extent(2)            
            , domain.prim.extent(3)            
        );

        // Solve loop
        do {
            aether::phys::set_dt(sim);
            sim.time.step++;
            std::cout << "The time step is " << sim.time.dt
                      << " The current time is " << sim.time.t << "\n";

            // Deep copy of the prims array 
            Kokkos::deep_copy(p_copy,domain.prim);
            Time_stepper(sim);
            boundary_conditions(sim, domain.cons);

            // Try the linear correction for the contact wave problem
            Kokkos::deep_copy(domain.prim,p_copy);
            if constexpr (aether::prim::Prim::HAS_VY) {
                correct_domain(sim);
            }
            aether::phys::cons_to_prims_domain(sim);

            if (sim.cfg.snap_shot_interval > 0 && sim.time.step % sim.cfg.snap_shot_interval == 0){
                aether::io::write_snapshot(sim, snap);
            }

        } while (sim.time.t < sim.time.t_end ); 

        std::cout << "The final time is " << sim.time.t << "\n";
        aether::io::write_snapshot(sim, snap);

    }; // namespace aether::core

    Kokkos::finalize();
    return 0;
}
