#pragma once
#include <aether/core/enums.hpp>
#include <string>

namespace aether::core {
    struct config{
        int x_count{0}, y_count{0}, z_count{0};
        double cfl{0.0}, t_end{0.0}, t_start{0.0};
        double x_start{0.0}, x_end{0.0};
        double y_start{0.0}, y_end{0.0};
        double z_start{0.0}, z_end{0.0};
        int snap_shot_interval{0};
        int num_ghost{0};
        int num_quad{0};
        test_problem prob;
        riemann riem = riemann::hll;
        solver solve = solver::fog;
        time_stepper time_step = time_stepper::char_trace;
        bool write_text{false}, write_ascii{false}, write_hdf5{false};
        bool use_defaults;
        std::string output_dir;
    };
}