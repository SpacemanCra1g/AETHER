#include <aether/core/RunParams.hpp>
#include <aether/core/RunParams_io.hpp>


int main(){
  using namespace aether::core; {

  Config cfg; 

  load_run_parameters(cfg);
  // display_run_parameters(cfg);

  
  
  }; // namespace aether::core
  return 0;
}
