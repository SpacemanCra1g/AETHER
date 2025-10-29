#pragma once 
#include <aether/core/config_build.hpp>

#if AETHER_PHYSICS_EULER==1
    #include <aether/physics/euler/public.hpp>
#elif AETHER_PHYSICS_MHD==1
    #include <aether/physics/mhd/public.hpp>
#elif AETHER_PHYSICS_SRHD==1
    #include <aether/physics/srhd/public.hpp>
#endif

namespace aether{
#if AETHER_PHYSICS_EULER==1
    namespace phys = ::aether::physics::euler;

#elif AETHER_PHYSICS_MHD==1
    namespace phys = physics::mhd;

#elif AETHER_PHYSICS_SRHD==1
    namespace phys = physics::srhd;
#endif
}