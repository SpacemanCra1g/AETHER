#include "aether/core/config.hpp"
#include "aether/core/config_build.hpp"
#include <aether/io/snapshot.hpp>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <filesystem>
#include <aether/core/simulation.hpp>

namespace aether::io {

struct index_ranges {
    int i0{0}, i1{0}, j0{0}, j1{0};
    int k0{0}, k1{0};
};    

static AETHER_INLINE int num_digits(int value){
    if (value <= 0) return 2;
    int digits = 0;
    while (value) {
        value /= 10;
        ++digits;
    }
    return digits;
}

static AETHER_INLINE std::string make_snapshot_path(const std::string &dir,
                                      const std::string &prefix,
                                      const int step, const std::string &ext = ".dat"){

    namespace fs = std::filesystem;
        
    // create the directory 
    fs::create_directory(dir);

    // create trailing step number, width 6 with padding leading 0's
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << step;
    // create the file name by joining the following together: eg "snap_000110.txt"
    const std::string f_name = prefix + "_" + oss.str() + ext;

    // OS agnostic fs operator / joins directories with filenames according to OS standard 
    // i.e. "\" on windows, and "/" everywhere else :) 
    return (fs::path(dir) / f_name).string();

}

static AETHER_INLINE void write_plaintext_header(FILE* f,
                                          const aether::core::Simulation& sim){
    const auto& g = sim.grid;
    const auto& t = sim.time;

    std::fprintf(f, "# AETHER snapshot (PlainText)\n");
    std::fprintf(f, "# dim=%d  numvar=%d  step=%d\n",
                 AETHER_DIM, aether::core::Simulation::numvar, t.step);
    std::fprintf(f, "# t=%.16e  dt=%.16e  cfl=%.6f\n", t.t, t.dt, t.cfl);

    // Base grid info (always has x)
    std::fprintf(f, "# nx=%d", g.nx);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "  ny=%d", g.ny);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "  nz=%d", g.nz);
    std::fprintf(f, "  ng=%d  quad=%d  gamma=%.6f\n",
                 g.ng, g.quad, g.gamma);

    // Domain extents
    std::fprintf(f, "# domain: x[%.6f, %.6f]", g.x_min, g.x_max);
    if constexpr (AETHER_DIM >= 2)
        std::fprintf(f, "  y[%.6f, %.6f]", g.y_min, g.y_max);
    if constexpr (AETHER_DIM == 3)
        std::fprintf(f, "  z[%.6f, %.6f]", g.z_min, g.z_max);
    std::fprintf(f, "\n#\n");
}

static AETHER_INLINE void write_plaintext_column_header(FILE* f){
    std::fprintf(f,
        "# Variable order: indices (i[,j[,k]]) then state variables v0..v%d\n",
        aether::core::Simulation::numvar - 1);

    std::fprintf(f, "#");
    if constexpr (AETHER_DIM == 1) {
        std::fprintf(f, "  i");
    } else if constexpr (AETHER_DIM == 2) {
        std::fprintf(f, "  i  j");
    } else if constexpr (AETHER_DIM == 3) {
        std::fprintf(f, "  i  j  k");
    }

    for (int v = 0; v < aether::core::Simulation::numvar; ++v) {
        std::fprintf(f, "  v%d", v);
    }
    std::fprintf(f, "\n");
}

static AETHER_INLINE index_ranges build_index_ranges(aether::core::Simulation &sim, bool include_ghosts=true){
    index_ranges R{}; 
    auto &g = sim.grid;
    auto ng = g.ng;

    if (include_ghosts) {R.i0 = -ng; R.i1 = g.nx + ng;}
    else {R.i0 = 0; R.i1 = g.nx;}

    if constexpr (AETHER_DIM > 1) {
        if (include_ghosts) {R.j0 = -ng; R.j1 = g.ny + ng;}
        else {R.j0 = 0; R.j1 = g.ny;}
    }

    if constexpr (AETHER_DIM == 3) {
        if (include_ghosts) {R.k0 = -ng; R.k1 = g.nz + ng;}
        else {R.k0 = 0; R.k1 = g.nz;}
    }
    return R;
}

static AETHER_INLINE void write_plain_text_cell_line(FILE* f, 
    aether::core::Simulation::View& view, int i, int j = 0, int k = 0){

    int i_width = num_digits(view.nx+view.ng*2);  
    int k_width = 0;
    int j_width = 0;
    
    #if AETHER_DIM > 1
        j_width = num_digits(view.ny+view.ng*2);
    #elif AETHER_DIM > 2
        k_width = num_digits(view.nz+view.ng*2);
    #endif  

    if constexpr (AETHER_DIM == 1) std::fprintf(f, "%*d", i_width, i);
    else if constexpr (AETHER_DIM == 2) std::fprintf(f, "%*d  %*d", i_width, i, j_width, j);
    else if constexpr (AETHER_DIM == 3)std::fprintf(f, "%*d  %*d  %*d", i_width, i, j_width, j, k_width, k);
    
    for (int c = 0; c < aether::core::Simulation::numvar; ++c) std::fprintf(f, "  %.16e", view.prim.var(c,i,j,k));

    std::fprintf(f, "\n");

}

static AETHER_INLINE void dump_plaintext_prims_rows(
    FILE* f,  aether::core::Simulation& sim, bool include_ghosts){

    auto V = sim.view();                
    const auto R = build_index_ranges(sim, include_ghosts);

    if constexpr (AETHER_DIM == 1) {
        for (int i = R.i0; i < R.i1; ++i)
            write_plain_text_cell_line(f, V, i);
    } else if constexpr (AETHER_DIM == 2) {
        for (int j = R.j0; j < R.j1; ++j)
            for (int i = R.i0; i < R.i1; ++i)
                write_plain_text_cell_line(f, V, i, j);
    } else { // AETHER_DIM == 3
        for (int k = R.k0; k < R.k1; ++k)
            for (int j = R.j0; j < R.j1; ++j)
                for (int i = R.i0; i < R.i1; ++i)
                    write_plain_text_cell_line(f, V, i, j, k);
    }
}

static AETHER_INLINE void write_plaintext_snapshot(
    aether::core::Simulation& sim,
    const std::string& outdir,
    const std::string& prefix,
    bool include_ghosts){

    const std::string path = make_snapshot_path(outdir, prefix, sim.time.step, ".dat");

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { throw std::runtime_error("Failed to open " + path); }

    // Small metadata header (the short function we did earlier)
    write_plaintext_header(f, sim);

    // Column labels: "# i [j [k]] v0 v1 ..."
    write_plaintext_column_header(f);

    // Rows: one line per cell (prims)
    dump_plaintext_prims_rows(f, sim, include_ghosts);

    std::fclose(f);
}

void write_snapshot(aether::core::Simulation &Sim, snapshot_request &req){
    for (output_format type : req.formats){
        if (type == output_format::plain_txt){
            write_plaintext_snapshot(Sim,req.output_dir,req.prefix,req.include_ghosts);
        }
    }
}

} // namespace: aether::io