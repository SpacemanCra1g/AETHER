#include "aether/core/enums.hpp"
#include <aether/core/enums_util.hpp>

namespace aether::core {

std::string_view to_string(riemann r){
    switch(r){
        case(riemann::hll) : return "hll";
        case(riemann::hllc) : return "hllc";
        case(riemann::roe) : return "roe";
        case(riemann::exact) : return "exact";
        default: return "unknown";
    }
}

std::string_view to_string(solver r){
    switch(r){
        case(solver::fog) : return "fog";
        case(solver::plm) : return "plm";
        case(solver::ppm) : return "ppm";
        case(solver::weno3) : return "weno3";
        case(solver::weno5) : return "weno5";
        default: return "unknown";
    }
}

std::string_view to_string(time_stepper r){
    switch(r){
        case(time_stepper::char_trace) : return "characteristic trace";
        case(time_stepper::rk1) : return "forward euler";
        case(time_stepper::rk2) : return "ssp-rk2";
        case(time_stepper::rk3) : return "ssp-rk3";
        case(time_stepper::rk4) : return "ssp-rk4";
        default: return "unknown";
    }
}

std::string_view to_string(test_problem r){
    switch(r){
        case(test_problem::custom) : return "Custom Test Problem";
        case(test_problem::dmr) : return "DMR Test Problem";
        case(test_problem::load) : return "Loading Test Problem";
        case(test_problem::sedov) : return "Sedov Test Problem";
        case(test_problem::sod) : return "Sod ShockTube Test Problem";
        case(test_problem::sr_shocktube) : return "SRHD ShockTube Test Problem";
        default: return "unknown";
    }
}

std::string_view to_string(boundary_conditions r){
    switch(r){
        case(boundary_conditions::Outflow) : return "Outflow Boundary Conditions";
        case(boundary_conditions::Periodic) : return "Periodic Boundary Conditions";
        case(boundary_conditions::Reflecting) : return "Reflecting Boundary Conditions";
        default: return "unknown";
    }
}
}