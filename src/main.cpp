#include "aether/core/enums.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/TemporalDispatch.hpp>
#include <fenv.h>

int main() {
    feenableexcept(FE_INVALID);
    using namespace aether::core;
    {
        Kokkos::initialize();

        Config cfg;
        Simulation sim;

        load_run_parameters(cfg);
        check_run_parameters(cfg);
        sim = Simulation(cfg);

        initialize_domain(sim);
        auto domain = sim.view();        

        // This is really inefficient and bad. Learn to live with it, DMR is a hell of a drug
        boundary_conditions(sim,domain.prim);
        aether::phys::prims_to_cons_domain(sim);
        boundary_conditions(sim,domain.cons);        

        do {
            aether::phys::set_dt(sim);
            std::cout << "The time step is " << sim.time.dt
                      << " The current time is " << sim.time.t << "\n";

            Time_stepper(sim);
            boundary_conditions(sim, domain.cons);
            aether::phys::cons_to_prims_domain(sim);
        } while (sim.time.t < sim.time.t_end ); 
    
        std::cout << "The final time is " << sim.time.t << "\n";
    
        aether::io::snapshot_request snap;
        snap.formats.push_back(aether::io::output_format::plain_txt);
        snap.output_dir = "Output";
        snap.prefix = "snap";
        snap.include_ghosts = true;

        aether::io::write_snapshot(sim, snap);

    }; // namespace aether::core

    Kokkos::finalize();
    return 0;
}
