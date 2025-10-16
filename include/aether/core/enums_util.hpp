#pragma once
#include <string_view>
#include <aether/core/enums.hpp>

namespace aether::core {

std::string_view to_string(riemann R);

std::string_view to_string(solver R);

std::string_view to_string(time_stepper R);

std::string_view to_string(test_problem R);

// inline constexpr std::string_view to_string(aether::core::OutputType R);

}