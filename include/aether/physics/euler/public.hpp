#pragma once
#include <string_view>
#include <aether/physics/euler/convert.hpp>
#include <aether/physics/euler/time_controller.hpp>

namespace aether::physics::euler{

int nprim();
int ncons();
std::string_view name();


}