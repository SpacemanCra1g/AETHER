#include "aether/physics/euler/convert.hpp"
#include "impl/Kokkos_InitializeFinalize.hpp"
#include <aether/core/simulation.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/time_stepper_containers.hpp>
#include <aether/core/SpaceDispatch.hpp>
#include <aether/core/RiemannDispatch.hpp>
#include <aether/core/flux_difference.hpp>
#include <aether/core/CTU/ctu_total_correction.hpp>
#include <aether/io/snapshot.hpp>
#include <omp.h>
#include <fenv.h>
#include <Kokkos_Core.hpp>

int main(){
  feenableexcept(FE_INVALID);
  using namespace aether::core; {
  
  Kokkos::initialize();

  Config cfg; 
  Simulation sim;

  load_run_parameters(cfg);
  sim = Simulation(cfg);
  
  initialize_domain(sim);
  // Initialize buffers
  // substage_container buffers;
  // buffers.init(sim);
  
  auto domain = sim.view();
  boundary_conditions(sim, domain.prim);
  
  aether::phys::prims_to_cons_domain(sim);
  // aether::phys::cons_to_prims_domain(sim);

  std::cout << "Here is the physics name: " << aether::phys::name() << "\n";

  std::cout << "The max signal speed is: " << aether::phys::max_propagation_speed(sim) << "\n";

  // std::cout << sim.time.t << " Start";

    do {
    aether::phys::set_dt(sim);
    std::cout << "The time step is " << sim.time.dt << " The current time is " << sim.time.t << "\n";

    Space_solve(sim);
    CTU_correction(sim);

    Riemann_dispatch(sim,domain);

    flux_diff_sweep(domain.prim, sim);
    axpy(domain.cons, -1.0, domain.prim);
    boundary_conditions(sim,domain.cons);
    aether::phys::cons_to_prims_domain(sim);
  } while (sim.time.t < sim.time.t_end);
    // aether::phys::calc_eigenvecs(View.prim, View.eigs, sim.grid.gamma);
    
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
