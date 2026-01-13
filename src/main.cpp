// #include "aether/physics/euler/pop_eigs.hpp"
// #include "aether/physics/euler/time_controller.hpp"
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
  std::cout << "\nThe time step is " << sim.time.dt << "\n";
  
  aether::io::snapshot_request snap;
  snap.formats.push_back(aether::io::output_format::plain_txt);
  snap.output_dir = "Output";
  snap.prefix = "snap";
  snap.include_ghosts = true;

  aether::io::write_snapshot(sim, snap);

  // Now we see if we have created the spectral decomposition correctly 

  // First step, populate the eigenvectors 
    // eigenvectors char_eigs;
    // eigenvec_view eigs;

    std::cout << "Did I crash?? \n";
    aether::phys::calc_eigenvecs(View.prim, View.eigs, sim.grid.gamma);
    std::cout << sim.char_eigs.x_eigs[45][2] << "\n";
    std::cout << View.eigs.x_eigs[45][2] << "\n";

  // CharSoA chars_container;
  // CharView chars;
  
  }; // namespace aether::core
  return 0;
}
