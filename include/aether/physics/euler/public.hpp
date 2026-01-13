#pragma once
#include <string_view>
#include <aether/physics/euler/convert.hpp>
#include <aether/physics/euler/time_controller.hpp>
#include <aether/physics/euler/pop_eigs.hpp>
#include <aether/physics/euler/variable_structs.hpp>

namespace aether::physics::euler{

int nprim();
int ncons();
std::string_view name();


}