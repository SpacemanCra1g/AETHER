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
    // feenableexcept(FE_INVALID);
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

        // aether::core::one_cell_spectral_container chars;
        // aether::math::Vec<aether::phys_ct::numvar> vec;

        // vec.data[0] = domain.prim(0,0,50,50);
        // vec.data[1] = domain.prim(1,0,50,50) + 1.2;        
        // vec.data[2] = domain.prim(2,0,50,50);        
        // vec.data[3] = domain.prim(3,0,50,50);   
        // vec.data[4] = domain.prim(4,0,50,50);   
        
        // aether::phys::prims P;
        // P.rho = domain.prim(0,0,50,50);
        // P.vx = domain.prim(1,0,50,50) + 1.2;
        // P.vy = domain.prim(2,0,50,50);
        // P.vz = domain.prim(3,0,50,50);
        // P.p = domain.prim(4,0,50,50);

        // aether::physics::euler::fill_eigenvectors(P, chars, domain.gamma);

        // std::cout << std::sqrt(domain.gamma*P.p/P.rho) << "\n";
        // for (int i = 0; i < 5; ++i) std::cout << chars.y.eigs[i] << " ";
        // std::cout << "\n";
        // auto w = chars.y.left * vec;
        // for (int i = 0; i < 5; ++i) std::cout << w[i] << " ";
        // std::cout << "\n";
        // auto q = chars.y.right * w;
        // for (int i = 0; i < 5; ++i) std::cout << q[i] << " ";
        // std::cout << "\n";

        int count = 0;
        do {
            count++;
            aether::phys::set_dt(sim);
            std::cout << "The time step is " << sim.time.dt
                      << " The current time is " << sim.time.t << "\n";

            Time_stepper(sim);
            boundary_conditions(sim, domain.cons);
            aether::phys::cons_to_prims_domain(sim);
        } while (sim.time.t < sim.time.t_end && count < 2);

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
