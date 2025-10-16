#include <aether/core/RunParams.hpp>
#include <aether/core/config_build.hpp>
#include <aether/core/RunParams_io.hpp>
#include <aether/core/enums_util.hpp>
#include <ostream>


namespace aether::core {

void display_run_parameters(config &cfg,std::ostream& os){

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

}
}

