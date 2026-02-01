#include "aether/physics/euler/convert.hpp"
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
#include <omp.h>
int main(){
  using namespace aether::core; {

  Config cfg; 
  Simulation sim;

  load_run_parameters(cfg);
  sim = Simulation(cfg);
  
  initialize_domain(sim);
  // Initialize buffers
  substage_container buffers;
  buffers.init(sim);
  
  auto View = sim.view();
  boundary_conditions(sim,View.prim);
  
  aether::phys::prims_to_cons_domain(sim);
  // aether::phys::cons_to_prims_domain(sim);

  std::cout << "Here is the physics name: " << aether::phys::name() << "\n";

  std::cout << "The max signal speed is: " << aether::phys::max_propagation_speed(sim) << "\n";

  // std::cout << sim.time.t << " Start";

  int count = 0;
  while (sim.time.t < sim.time.t_end && count < 4){
    count ++;
    aether::phys::set_dt(sim);
    std::cout << "\nThe time step is " << sim.time.t << "\n";

    Space_solve(sim);
    Riemann_dispatch(sim, sim.grid.gamma);

    flux_diff_sweep(View.prim, sim);

    axpy(sim.cons_container, -1.0, sim.prims_container);
    boundary_conditions(sim,View.cons);
    aether::phys::cons_to_prims_domain(sim);
  }
    // aether::phys::calc_eigenvecs(View.prim, View.eigs, sim.grid.gamma);

  aether::io::snapshot_request snap;
  snap.formats.push_back(aether::io::output_format::plain_txt);
  snap.output_dir = "Output";
  snap.prefix = "snap";
  snap.include_ghosts = true;

  aether::io::write_snapshot(sim, snap);

    // std::cout << View.chars.var(1,2,100,100,0);
  
  }; // namespace aether::core
  return 0;
}
