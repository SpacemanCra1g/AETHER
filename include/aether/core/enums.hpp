#pragma once
#include <cstdint>

namespace aether::core {
    enum class riemann : std::uint8_t {hll=0,hllc=1,roe=2,exact=3};
    enum class solver : std::uint8_t {fog=0,plm=1,ppm=2,weno3=3,weno5=4};
    enum class time_stepper : std::uint8_t {char_trace=0,rk1=1,rk2=2,rk3=3,rk4=4};
    enum class test_problem : std::uint8_t {sod=0,sedov=1,sr_shocktube=2,dmr=3, custom=4, load=5, sod_y=6};
    enum class boundary_conditions : std::uint8_t {Outflow = 1, Periodic = 2, Reflecting = 3};
    // enum class OutputType : std::uint8_t {text=0,ascii=1,hdf5=2};
    enum class sweep_dir : std::uint8_t {x=0,y=1,z=2};
}