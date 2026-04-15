#include "Kokkos_Core_fwd.hpp"
#include "aether/core/enums.hpp"
#include "aether/physics/euler/convert.hpp"
#include "impl/Kokkos_InitializeFinalize.hpp"
#include <Kokkos_Core.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/core/simulation.hpp>
#include <aether/core/time_stepper_containers.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/TemporalDispatch.hpp>
#include <aether/core/char_struct.hpp>
#include <fenv.h>
#include <omp.h>

int main() {
    feenableexcept(FE_INVALID);
    using namespace aether::core;
    {
        Kokkos::initialize();

        Config cfg;
        Simulation sim;

        load_run_parameters(cfg);
        sim = Simulation(cfg);

        initialize_domain(sim);

        auto domain = sim.view();
        boundary_conditions(sim, domain.prim);

        aether::phys::prims_to_cons_domain(sim);


        int count = 0;
    
        do {
            count++;
            aether::phys::set_dt(sim);
            std::cout << "The time step is " << sim.time.dt
                      << " The current time is " << sim.time.t << "\n";

            Time_stepper(sim);
            boundary_conditions(sim, domain.cons);
            aether::phys::cons_to_prims_domain(sim);
        } while (sim.time.t < sim.time.t_end ); 
    
        std::cout << "The final time is " << sim.time.t << "\n";
    //     boundary_conditions(sim,domain.cons);
    //     boundary_conditions(sim,domain.prim);
    //     boundary_conditions(sim,sim.view().prim);
    //     Kokkos::fence();


    //     auto v = sim.view().prim;
    // auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);

    // int j;
    // const int ib = sim.cells.ibegin();
    // const int ie = sim.cells.iend();

    // for (j = 100; j < 104; j++){
    // std::cout << "j = " << j << "\n";
    // std::cout << "ib=" << ib << " ie=" << ie << "\n";
    // std::cout << "srcL  " << h(P::RHO,0,j,ib)   << "\n";
    // std::cout << "gL0   " << h(P::RHO,0,j,0)    << "\n";
    // std::cout << "gL1   " << h(P::RHO,0,j,1)    << "\n";
    // std::cout << "gL2   " << h(P::RHO,0,j,2)    << "\n";
    // std::cout << "srcR  " << h(P::RHO,0,j,ie-1) << "\n";
    // std::cout << "gR0   " << h(P::RHO,0,j,ie)   << "\n";
    // std::cout << "gR1   " << h(P::RHO,0,j,ie+1) << "\n";
    // std::cout << "gR2   " << h(P::RHO,0,j,ie+2) << "\n";
    // std::cout << "\n";

    // }


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
