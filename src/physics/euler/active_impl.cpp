#include <aether/physics/euler/public.hpp>
#include <aether/physics/counts.hpp>

namespace physics::euler{
    namespace ct = aether::phys_ct;
    int nprim(){return ct::numvar;}
    int ncons(){return ct::numvar;}
    std::string_view name(){return "Euler";}
}
