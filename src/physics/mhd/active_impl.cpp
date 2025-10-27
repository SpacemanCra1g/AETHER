#include <aether/physics/mhd/public.hpp>
#include <aether/physics/counts.hpp>

namespace physics::mhd{
    namespace ct = aether::phys_ct;
    int nprim(){return ct::numvar;}
    int ncons(){return ct::numvar;}
    std::string_view name(){return "MHD";}
}
