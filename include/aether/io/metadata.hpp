#pragma once
#include <aether/core/config.hpp>
#include <aether/io/snapshot.hpp>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace aether::io {

static AETHER_INLINE void prepare_output_directory(const snapshot_request& req) {
    namespace fs = std::filesystem;

    const fs::path outdir(req.output_dir);
    fs::create_directories(outdir);

    for (const auto& entry : fs::directory_iterator(outdir)) {
        if (!entry.is_regular_file()) continue;

        const auto name = entry.path().filename().string();

        // Remove this run's metadata file
        if (name == req.prefix + "_metadata.txt") {
            fs::remove(entry.path());
            continue;
        }

        // Remove this run's snapshot files
        const std::string stem_prefix = req.prefix + "_";
        if (name.rfind(stem_prefix, 0) == 0) {
            const auto ext = entry.path().extension().string();
            if (ext == ".dat" || ext == ".bin") {
                fs::remove(entry.path());
            }
        }
    }
}

static AETHER_INLINE std::string make_metadata_path(const std::string& dir, const std::string& prefix) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    return (fs::path(dir) / (prefix + "_metadata.txt")).string();
}

static AETHER_INLINE void write_run_metadata(const aether::core::Simulation& sim,
                                             const snapshot_request& req) {

    prepare_output_directory(req);

    const std::string path = make_metadata_path(req.output_dir, req.prefix);


    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }

    const auto& g = sim.grid;

    std::fprintf(f, "# AETHER run metadata\n");
    std::fprintf(f, "format = AETHER_SNAPSHOT_META\n");
    std::fprintf(f, "version = 1\n\n");

    std::fprintf(f, "snapshot_prefix = %s\n", req.prefix.c_str());
    std::fprintf(f, "snapshot_text_extension = .dat\n");
    std::fprintf(f, "snapshot_binary_extension = .bin\n\n");

    std::fprintf(f, "dimension = %d\n", AETHER_DIM);
    std::fprintf(f, "numvar = %d\n", aether::core::Simulation::numvar_full);
    std::fprintf(f, "scalar_type = double\n");
    std::fprintf(f, "payload_kind = primitive\n");
    std::fprintf(f, "file_order_binary = (var,k,j,i)\n");
    std::fprintf(f, "file_order_text = row-major with explicit indices\n");
    std::fprintf(f, "include_ghosts_default = %s\n",
                 req.include_ghosts ? "true" : "false");
    std::fprintf(f, "output_plain_txt = %s\n",
                 sim.cfg.write_text ? "true" : "false");                 
    std::fprintf(f, "output_binary = %s\n\n",
                 sim.cfg.write_binary ? "true" : "false");                                  

    std::fprintf(f, "nx = %d\n", g.nx);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "ny = %d\n", g.ny);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "nz = %d\n", g.nz);
    std::fprintf(f, "ng = %d\n\n", g.ng);

    std::fprintf(f, "x_min = %.16e\n", g.x_min);
    std::fprintf(f, "x_max = %.16e\n", g.x_max);
    if constexpr (AETHER_DIM >= 2) {
        std::fprintf(f, "y_min = %.16e\n", g.y_min);
        std::fprintf(f, "y_max = %.16e\n", g.y_max);
    }
    if constexpr (AETHER_DIM == 3) {
        std::fprintf(f, "z_min = %.16e\n", g.z_min);
        std::fprintf(f, "z_max = %.16e\n", g.z_max);
    }
    std::fprintf(f, "\n");

    std::fprintf(f, "dx = %.16e\n", g.dx);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "dy = %.16e\n", g.dy);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "dz = %.16e\n", g.dz);

    std::fprintf(f, "gamma = %.16e\n", g.gamma);
    std::fprintf(f, "quad = %d\n", g.quad);

    std::fclose(f);
}

}