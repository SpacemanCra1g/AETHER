#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <aether/core/simulation.hpp>

namespace aether::io {

    enum class output_format {plain_txt, binary};
    struct snapshot_request{
        std::vector<output_format> formats;
        std::string output_dir{""};
        std::string prefix{"snap"};
        bool include_ghosts{true};
    };

    struct SnapshotHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t step;      // timestep
    double   time;      // simulation time
    uint64_t payload_bytes;
};

struct index_ranges {
    int i0{0}, i1{0};
    int j0{0}, j1{0};
    int k0{0}, k1{0};
};

static AETHER_INLINE int num_digits(int value) {
    if (value <= 0) return 2;
    int digits = 0;
    while (value) {
        value /= 10;
        ++digits;
    }
    return digits;
}

static AETHER_INLINE index_ranges build_index_ranges(const aether::core::Simulation& sim,
                                                     bool include_ghosts = true) {
    index_ranges r{};
    const auto& g = sim.grid;
    const int ng = g.ng;

    if (include_ghosts) {
        r.i0 = -ng;      r.i1 = g.nx + ng;
    } else {
        r.i0 = 0;        r.i1 = g.nx;
    }

    if constexpr (AETHER_DIM > 1) {
        if (include_ghosts) {
            r.j0 = -ng;  r.j1 = g.ny + ng;
        } else {
            r.j0 = 0;    r.j1 = g.ny;
        }
    } else {
        r.j0 = 0;        r.j1 = 1;
    }

    if constexpr (AETHER_DIM == 3) {
        if (include_ghosts) {
            r.k0 = -ng;  r.k1 = g.nz + ng;
        } else {
            r.k0 = 0;    r.k1 = g.nz;
        }
    } else {
        r.k0 = 0;        r.k1 = 1;
    }
    return r;
}

static AETHER_INLINE std::string make_snapshot_path(const std::string& dir,
                                                    const std::string& prefix,
                                                    int step,
                                                    const std::string& ext = ".dat") {
    namespace fs = std::filesystem;

    fs::create_directories(dir);
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << step;

    const std::string fname = prefix + "_" + oss.str() + ext;
    return (fs::path(dir) / fname).string();
}
    
void write_snapshot(aether::core::Simulation &Sim, snapshot_request &req);
}