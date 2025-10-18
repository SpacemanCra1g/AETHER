#pragma once 
#include <iosfwd>
#include <iostream>

namespace aether::core {

struct Config;
void display_run_parameters(Config &cfg,std::ostream& os = std::cout);
void load_run_parameters(Config& cfg);
}