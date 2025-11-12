#include "aether/core/enums.hpp"
#include <aether/core/RunParams.hpp>
#include <aether/core/config_build.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/enums_util.hpp>

#include <cassert>
#include <fstream>

[[maybe_unused]] static bool print_if_false (std::ifstream &assertion, std::string Message){
  if (!assertion) {std::cout << Message << std::endl; return false;}
  else return true;
}

static void strip_comments(std::string &s){
    // Strips all comments from the config file
    int num = s.find("#"); 
    if (num >= 0) s.erase(s.begin() + num ,s.end());
}

static void strip_whitespace(std::string &s){
  std::string new_line;
    for (int it = 0 ; it < (int)s.length();it++) {
      if (s.at(it) == '\t' || s.at(it) == ' ' );
      else new_line += s.at(it);
    }
    s = std::move(new_line);
}

static bool load_domain_size(std::string &s, aether::core::Config &cfg){
  switch (s.at(4)) {
  case 'x' : cfg.x_count = std::stoi(s.substr(11,s.length()-11)); return true;

  case 'y' : cfg.y_count = std::stoi(s.substr(11,s.length()-11)); 
  if constexpr (AETHER_DIM < 2) cfg.y_count = 0;
  return true;

  case 'z' : cfg.z_count = std::stoi(s.substr(11,s.length()-11)); 
  if constexpr (AETHER_DIM < 3) cfg.z_count = 0;
  return true;


  default: return false;
  };
}

static bool load_domain_range(std::string &s, aether::core::Config &cfg){
  switch (s.at(0)) {
  case 'x' : 
    cfg.x_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.x_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  case 'y' : 
    cfg.y_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.y_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    if constexpr (AETHER_DIM < 2) {cfg.y_start = 0.0; cfg.y_end = 0.0;}
    return true;
  case 'z' : 
    cfg.z_start = std::stod(s.substr(27, s.find_last_of(':') - 27)); 
    cfg.z_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    if constexpr (AETHER_DIM < 3) {cfg.z_start = 0.0; cfg.z_end = 0.0;}
    return true;
  default: return false;
  };
}

static bool load_run_specification(std::string &s, aether::core::Config &cfg){
  if (s.substr(0,10) == "cfl_number") {
    cfg.cfl = std::stod(s.substr(10, s.length() - 10)); 
    return true;
  }
  else if (s.substr(0,10) == "time_start"){
    cfg.t_start = std::stod(s.substr(19, s.find_last_of(':') - 19)); 
    cfg.t_end = std::stod(s.substr( s.find_last_of(':')+1,s.length() - (s.find_last_of(':')+1))); 
    return true;
  }
  else if (s.substr(0,12) == "time_stepper"){
    std::string stepper = s.substr(12, s.length() - 12); 
    std::string stepper_lower; 
    for (std::size_t i = 0; i < stepper.length(); ++i){
       stepper_lower += std::tolower(stepper[i]);
    }
    if (stepper_lower == "char_trace"){
      cfg.time_step = aether::core::time_stepper::char_trace;
    }
    else if (stepper_lower == "rk1"){
      cfg.time_step = aether::core::time_stepper::rk1;
    }
    else if (stepper_lower == "rk2"){
      cfg.time_step = aether::core::time_stepper::rk2;
    }
    else if (stepper_lower == "rk3"){
      cfg.time_step = aether::core::time_stepper::rk3;
    }
    else if (stepper_lower == "rk4"){
      cfg.time_step = aether::core::time_stepper::rk4;
    }
    return true;
  }
  else if (s.substr(0,5) == "gamma"){
    cfg.gamma = std::stod(s.substr(5, s.length() - 5)); 
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
  else if (s.substr(0,19) == "boundary_conditions"){
    std::string BCs = s.substr(19, s.length() - 19); 
    std::string BCs_lower;
    for (std::size_t i = 0; i < BCs.length(); ++i){
       BCs_lower += std::tolower(BCs[i]);
    }
    if (BCs_lower == "outflow"){
       cfg.bc = aether::core::boundary_conditions::Outflow; return true;
    }
    else if (BCs_lower == "periodic"){
       cfg.bc = aether::core::boundary_conditions::Periodic; return true;
    }
    else if (BCs_lower == "reflecting"){
       cfg.bc = aether::core::boundary_conditions::Reflecting; return true;
    }
    else {
      throw std::runtime_error("Unknown Boundary Condition selection " + BCs_lower);
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
  else if (s.substr(0,15) == "num_ghost_cells"){
    cfg.num_ghost = std::stoi(s.substr(15, s.length() - 15)); 
    return true;
  }
  else{
  return false;
  }
};

static bool load_output_specification(std::string &s, aether::core::Config &cfg){
  if (s.substr(0,10) == "write_text"){
    std::string write_text = s.substr(10, s.length() - 10); 
    std::string write_text_lower;
    for (std::size_t i = 0; i < write_text.length(); ++i){
      write_text_lower += std::tolower(write_text[i]);
    }
    if (write_text_lower == "false") cfg.write_text = false;
    else if (write_text_lower == "true") cfg.write_text = true;
    else throw std::runtime_error("Unknown parameter in 'write_text'");
  }
  else if (s.substr(0,11) == "write_ascii"){
    std::string write_ascii = s.substr(11, s.length() - 11); 
    std::string write_ascii_lower;
    for (std::size_t i = 0; i < write_ascii.length(); ++i){
      write_ascii_lower += std::tolower(write_ascii[i]);
    }
    if (write_ascii_lower == "false") cfg.write_ascii = false;
    else if (write_ascii_lower == "true") cfg.write_ascii = true;
    else throw std::runtime_error("Unknown parameter in 'write_ascii'");
  }
  else if (s.substr(0,10) == "write_hdf5"){
    std::string write_ascii = s.substr(10, s.length() - 10); 
    std::string write_ascii_lower;
    for (std::size_t i = 0; i < write_ascii.length(); ++i){
      write_ascii_lower += std::tolower(write_ascii[i]);
    }
    if (write_ascii_lower == "false") cfg.write_hdf5 = false;
    else if (write_ascii_lower == "true") cfg.write_hdf5 = true;
    else throw std::runtime_error("Unknown parameter in 'write_hdf5'");
  }
  else if (s.substr(0,14) == "snapshot_every"){
    cfg.snap_shot_interval = std::stoi(s.substr(14,s.length()-14)); 
    return true;
  }
  else if (s.substr(0,17) == "custom_output_dir"){
  cfg.output_dir= s.substr(18,s.length()-19); 
  if (cfg.output_dir.empty()) cfg.output_dir = "OutputData/";
  return true;
  }

  return false;
};

namespace aether::core {

void display_run_parameters(Config& cfg,std::ostream& os){

    constexpr bool kCUDA   = (AETHER_ENABLE_CUDA   == 1);
    constexpr bool kMPI    = (AETHER_ENABLE_MPI    == 1);
    constexpr bool kOpenMP = (AETHER_ENABLE_OPENMP == 1);

    os << "=== AETHER run parameters ===\n";
    os << "physics           : " << AETHER_PHYSICS_STR << "\n";
    os << "cuda              : " << (kCUDA   ? "on" : "off") << "\n";
    os << "mpi               : " << (kMPI    ? "on" : "off") << "\n";
    os << "openmp            : " << (kOpenMP ? "on" : "off") << "\n";

    // from cfg
    os << "grid (nx,ny,nz)   : " << cfg.x_count << "," <<
        cfg.y_count << "," << cfg.z_count << "\n";
    os << "ghost cells       : " << cfg.num_ghost << "\n";
    os << "domain x          : [" << cfg.x_start << ", " << cfg.x_end << "]\n";
    if (cfg.y_count > 0) os << "domain y          : [" << cfg.y_start << ", " << cfg.y_end << "]\n";
    if (cfg.z_count > 0) os << "domain z          : [" << cfg.z_start << ", " << cfg.z_end << "]\n";
    os << "time              : [" << cfg.t_start << ", " << cfg.t_end << "]\n";
    os << "cfl number        : " << cfg.cfl <<"\n";
    os << "Riemann Solver    : " << to_string(cfg.riem) <<"\n";
    os << "Spatial Solver    : " << to_string(cfg.solve) <<"\n";
    os << "Quadrature points : " << cfg.num_quad <<"\n";
    os << "Test Problem      : " << to_string(cfg.prob) <<"\n";
    os << "Output type       : ";
    if (cfg.write_text) os << "plain text: ";
    if (cfg.write_ascii) os << "ascii file: ";
    if (cfg.write_hdf5) os << "hdf5 file: ";
    os << "\n";
    os << "Snapshot every    : " << cfg.snap_shot_interval << " time steps";
}

void load_run_parameters(Config& cfg){
    std::string line;
    std::ifstream in("./aether_config.cfg");
    assert(print_if_false(in,"Failed to open file"));


    while (std::getline(in,line)){

        // Strips all comments from the config file
        strip_comments(line);
        
        // Strips all whitespace from the config file
        strip_whitespace(line);

        // Do not move forward with empty strings
        if (line.empty()) continue;

        bool success = load_domain_size(line,cfg);
        // if (success) std::cout << "Domain size loaded!\n";

        success = load_domain_range(line,cfg);
        // if (success) std::cout << "Domain boundaries loaded!\n";

        success = load_run_specification(line,cfg);
        // if (success) std::cout << "Run Specifications loaded!\n";

        success = load_output_specification(line,cfg);
        // if (success) std::cout << "Output Specifications loaded!\n";
        (void)success;
    }
}
};