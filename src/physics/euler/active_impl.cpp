#include <aether/physics/api.hpp>
#include <aether/physics/counts.hpp>

namespace aether::phys{
    namespace ct = aether::phys_ct;
    int nprim(){return ct::numvar;}
    int ncons(){return ct::numvar;}
    std::string_view name(){return "Euler";}
}
