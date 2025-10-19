#include "aether/core/simulation.hpp"
#include <aether/core/RunParams.hpp>
#include <aether/core/RunParams_io.hpp>


int main(){
  using namespace aether::core; {

  Config cfg; 
  Simulation sim;

  load_run_parameters(cfg);
  sim = Simulation(cfg);
  // display_run_parameters(cfg);

  std::cout << "CFL = " << sim.time.cfl << "\n";
  std::cout << "time = " << sim.time.t << "\n";
  std::cout << "x_start = " << sim.grid.x_min << "\n";  
  std::cout << "x_end = " << sim.grid.x_max << "\n";  
  std::cout << "dx = " << sim.grid.dx << "\n";  
  std::cout << "nx = " << sim.grid.nx << "\n";  

  std::cout << "#####################\n y-Params\n";
  std::cout << "y_start = " << sim.grid.y_min << "\n";  
  std::cout << "y_end = " << sim.grid.y_max << "\n";  
  std::cout << "dy = " << sim.grid.dy << "\n";  
  std::cout << "ny = " << sim.grid.ny << "\n";  

  std::cout << "#####################\n z-Params\n";
  std::cout << "z_start = " << sim.grid.z_min << "\n";  
  std::cout << "z_end = " << sim.grid.z_max << "\n";  
  std::cout << "dz = " << sim.grid.dz << "\n";  
  std::cout << "nz = " << sim.grid.nz << "\n";  
  
  }; // namespace aether::core
  return 0;
}
