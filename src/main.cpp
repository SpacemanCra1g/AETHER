#include <aether/physics/api.hpp>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cassert>
#include <aether/core/RunParams.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/enums.hpp>

bool print_if_false (std::ifstream &assertion, std::string Message){
  if (!assertion) {std::cout << Message << std::endl; return false;}
  else return true;
}

void strip_comments(std::string &s){
    // Strips all comments from the config file
    int num = s.find("#"); 
    if (num >= 0) s.erase(s.begin() + num ,s.end());
}

void strip_whitespace(std::string &s){
  std::string new_line;
    for (int it = 0 ; it < (int)s.length();it++) {
      if (s.at(it) == '\t' || s.at(it) == ' ' );
      else new_line += s.at(it);
    }
    s = std::move(new_line);
}

bool load_domain_size(std::string &s, aether::core::config &cfg){
  switch (s.at(4)) {
  case 'x' : cfg.x_count = std::stoi(s.substr(11,s.length()-11)); return true;
  case 'y' : cfg.y_count = std::stoi(s.substr(11,s.length()-11)); return true;
  case 'z' : cfg.z_count = std::stoi(s.substr(11,s.length()-11)); return true;
  default: return false;
  };
}

bool load_domain_range(std::string &s, aether::core::config &cfg){
  switch (s.at(0)) {
  case 'x' : 
    cfg.x_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.x_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  case 'y' : 
    cfg.y_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.y_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  case 'z' : 
    cfg.z_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.z_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  default: return false;
  };
}

bool load_run_specification(std::string &s, aether::core::config &cfg){
  if (s.substr(0,10) == "cfl_number") {
    cfg.cfl = std::stod(s.substr(10, s.length() - 10)); 
    return true;
  }
  else if (s.substr(0,10) == "time_start"){
    cfg.t_start = std::stod(s.substr(19, s.find_last_of(':') - 19)); 
    cfg.t_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  }
  else if (s.substr(0,14) == "riemann_solver"){
    std::string riemann_solver = s.substr(14, s.length() - 14); 
    std::string riemann_solver_lower;
    for (std::size_t i = 0; i < riemann_solver.length(); ++i){
       riemann_solver_lower += std::tolower(riemann_solver[i]);
    }
    if (riemann_solver_lower == "hll"){
       cfg.riem = aether::core::riemann::hll; return true;
    }
    else if (riemann_solver_lower == "hllc"){
      cfg.riem = aether::core::riemann::hllc; return true;
    }
    else if (riemann_solver_lower == "roe"){
      cfg.riem = aether::core::riemann::roe; return true;
    }
    else if (riemann_solver_lower == "exact"){
      cfg.riem = aether::core::riemann::exact; return true;
    }
    else {
      throw std::runtime_error("Unknown Riemann Solver selection " + riemann_solver_lower);
      return false;
    }
  }
  else if (s.substr(0,14) == "spatial_solver"){
    std::string space_solver = s.substr(14, s.length() - 14); 
    std::string space_solver_lower;
    for (std::size_t i = 0; i < space_solver.length(); ++i){
       space_solver_lower += std::tolower(space_solver[i]);
    }
    if (space_solver_lower == "fog"){
       cfg.solve= aether::core::solver::fog; return true;
    }
    else if (space_solver_lower == "plm"){
      cfg.solve = aether::core::solver::plm; return true;
    }
    else if (space_solver_lower == "ppm"){
      cfg.solve = aether::core::solver::ppm; return true;
    }
    else if (space_solver_lower == "weno3"){
      cfg.solve= aether::core::solver::weno3; return true;
    }
    else if (space_solver_lower == "weno5"){
      cfg.solve= aether::core::solver::weno5; return true;
    }
    else {
      throw std::runtime_error("Unknown Space Solver selection " + space_solver_lower);
      return false;
    }
  }
  else if (s.substr(0,21) == "num_quadrature_points"){
    cfg.num_quad = std::stoi(s.substr(21,s.length()-21)); 
    return true;
  }
  else if (s.substr(0,12) == "test_problem"){
    std::string test_prob = s.substr(12, s.length() - 12); 
    std::string test_prob_lower;
    for (std::size_t i = 0; i < test_prob.length(); ++i){
      test_prob_lower += std::tolower(test_prob[i]);
    }
    if (test_prob_lower == "sod"){
      cfg.prob = aether::core::test_problem::sod; return true;
    }
    else if (test_prob_lower == "dmr"){
      cfg.prob = aether::core::test_problem::dmr; return true;
    }
    else if (test_prob_lower == "sedov"){
      cfg.prob = aether::core::test_problem::sedov; return true;
    }
    else if (test_prob_lower == "custom"){
      cfg.prob = aether::core::test_problem::custom; return true;
    }
    else if (test_prob_lower == "load"){
      cfg.prob = aether::core::test_problem::load; return true;
    }
    else {
      throw std::runtime_error("Unknown test problem selection " + test_prob_lower);
      return false;
    }
  }
  else if (s.substr(0,17) == "use_test_defaults"){
    std::string use_defaults = s.substr(17, s.length() - 17); 
    std::string use_defaults_lower;
    if (use_defaults_lower == "true") {cfg.use_defaults = true; return true;}
    else if (use_defaults_lower == "false") {cfg.use_defaults = false; return true;}
    return false;
  }

  else{
  return false;
  }
};



int main(){
  std::ifstream in("./aether_config.cfg");
  assert(print_if_false(in,"Failed to open file"));

  using namespace aether::core; {
  config cfg; 

  std::string line;
  unsigned int line_num = 0; 

  while (std::getline(in,line)){

    // Strips all comments from the config file
    strip_comments(line);
    
    // Strips all whitespace from the config file
    strip_whitespace(line);

    // Do not move forward with empty strings
    if (line.empty()) continue;
    line_num++;


    bool success = load_domain_size(line,cfg);
    if (success) std::cout << "Domain size loaded!\n";

    success = load_domain_range(line,cfg);
    if (success) std::cout << "Domain boundaries loaded!\n";

    success = load_run_specification(line,cfg);
    if (success) std::cout << "Run Specifications loaded!\n";

    

    
    // std::cout << line << std::endl;
  }; // while loop over .cfg file lines
  display_run_parameters(cfg);
  // std::cout << "Total line number: "  << line_num << std::endl;
  }; // namespace aether::core



  return 0;
}
