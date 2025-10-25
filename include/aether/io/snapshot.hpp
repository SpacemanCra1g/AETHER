#pragma once
#include <vector>
#include <string>
#include <aether/core/simulation.hpp>

namespace aether::io {

    enum class output_format {plain_txt, binary, hdf5};

    struct snapshot_request{
        std::vector<output_format> formats;
        std::string output_dir{""};
        std::string prefix{"snap"};
        bool include_ghosts{true};
    };

    void write_snapshot(aether::core::Simulation &Sim, snapshot_request &req);
}