#include <aether/physics/srhd/public.hpp>
#include <aether/physics/counts.hpp>

namespace aether::physics::srhd{
    namespace ct = aether::phys_ct;
    int nprim(){return ct::numvar;}
    int ncons(){return ct::numvar;}
    std::string_view name(){return "SRHD";}
}
