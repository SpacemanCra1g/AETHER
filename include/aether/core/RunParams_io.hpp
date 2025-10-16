#pragma once 
#include <iosfwd>
#include <iostream>

namespace aether::core {

struct config;
void display_run_parameters(config &cfg,std::ostream& os = std::cout);

}