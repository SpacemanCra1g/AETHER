#include <aether/core/simulation.hpp>
#include <aether/core/RunParams.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/Initialize.hpp>
#include <aether/core/boundary_conditions.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/physics/api.hpp>


int main(){
  using namespace aether::core; {

  Config cfg; 
  Simulation sim;

  load_run_parameters(cfg);
  sim = Simulation(cfg);

  initialize_domain(sim);
  auto View = sim.view();
  boundary_conditions(sim,View.prim);

  std::cout << "Here is the physics name: " << aether::phys::name() << "\n";
  
  
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
