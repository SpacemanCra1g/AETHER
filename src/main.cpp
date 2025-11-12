#include "aether/physics/counts.hpp"
#include "aether/physics/euler/time_controller.hpp"
#include <aether/core/simulation.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/time_stepper_containers.hpp>

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

  std::cout << "The max signal speed is: " << aether::phys::max_propagation_speed(sim);

  aether::phys::set_dt(sim);
  std::cout << "\nThe time step is " << sim.time.dt;
  
  namespace io = aether::io;
  io::snapshot_request snap;
  snap.formats.push_back(io::output_format::plain_txt);
  snap.output_dir = "Output";
  snap.prefix = "snap";
  snap.include_ghosts = true;

  io::write_snapshot(sim, snap);
  
  }; // namespace aether::core
  return 0;
}
