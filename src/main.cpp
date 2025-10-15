#include <aether/physics/api.hpp>
#include <iostream>

int main(){
 std::cout << "Number of Prims: " << aether::phys::nprim() << "\n"
           << "Number of Cons: " << aether::phys::ncons() << "\n"
           << "Physics Regime: "<< aether::phys::name() << std::endl;

  return 0;
}
