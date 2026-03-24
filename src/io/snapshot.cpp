#include <Kokkos_Core.hpp>

#include <aether/core/config.hpp>
#include <aether/core/config_build.hpp>
#include <aether/io/snapshot.hpp>

#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aether::io {

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

static AETHER_INLINE void write_plaintext_header(FILE* f,
                                                 const aether::core::Simulation& sim) {
    const auto& g = sim.grid;
    const auto& t = sim.time;

    std::fprintf(f, "# AETHER snapshot (PlainText)\n");
    std::fprintf(f, "# dim=%d  numvar=%d  step=%d\n",
                 AETHER_DIM, aether::core::Simulation::numvar, t.step);
    std::fprintf(f, "# t=%.16e  dt=%.16e  cfl=%.6f\n", t.t, t.dt, t.cfl);

    std::fprintf(f, "# nx=%d", g.nx);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "  ny=%d", g.ny);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "  nz=%d", g.nz);
    std::fprintf(f, "  ng=%d  quad=%d  gamma=%.6f\n",
                 g.ng, g.quad, g.gamma);

    std::fprintf(f, "# domain: x[%.6f, %.6f]", g.x_min, g.x_max);
    if constexpr (AETHER_DIM >= 2) {
        std::fprintf(f, "  y[%.6f, %.6f]", g.y_min, g.y_max);
    }
    if constexpr (AETHER_DIM == 3) {
        std::fprintf(f, "  z[%.6f, %.6f]", g.z_min, g.z_max);
    }
    std::fprintf(f, "\n#\n");
}

static AETHER_INLINE void write_plaintext_column_header(FILE* f) {
    std::fprintf(f,
        "# Variable order: indices (i[,j[,k]]) then state variables v0..v%d\n",
        aether::core::Simulation::numvar - 1);

    std::fprintf(f, "#");
    if constexpr (AETHER_DIM == 1) {
        std::fprintf(f, "  i");
    } else if constexpr (AETHER_DIM == 2) {
        std::fprintf(f, "  i  j");
    } else {
        std::fprintf(f, "  i  j  k");
    }

    for (int v = 0; v < aether::core::Simulation::numvar; ++v) {
        std::fprintf(f, "  v%d", v);
    }
    std::fprintf(f, "\n");
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

template<class HostView>
static AETHER_INLINE void write_plain_text_cell_line(FILE* f,
                                                     const HostView& prim_h,
                                                     const aether::core::Simulation& sim,
                                                     int i, int j = 0, int k = 0) {
    const int ng = sim.grid.ng;

    const int ii = i + ng;
    const int jj = (AETHER_DIM > 1) ? (j + ng) : 0;
    const int kk = (AETHER_DIM > 2) ? (k + ng) : 0;

    const int i_width = num_digits(sim.grid.nx + 2 * sim.grid.ng);
    const int j_width = (AETHER_DIM > 1) ? num_digits(sim.grid.ny + 2 * sim.grid.ng) : 0;
    const int k_width = (AETHER_DIM > 2) ? num_digits(sim.grid.nz + 2 * sim.grid.ng) : 0;

    if constexpr (AETHER_DIM == 1) {
        std::fprintf(f, "%*d", i_width, i);
    } else if constexpr (AETHER_DIM == 2) {
        std::fprintf(f, "%*d  %*d", i_width, i, j_width, j);
    } else {
        std::fprintf(f, "%*d  %*d  %*d", i_width, i, j_width, j, k_width, k);
    }

    for (int c = 0; c < aether::core::Simulation::numvar; ++c) {
        std::fprintf(f, "  %.16e", prim_h(c, kk, jj, ii));
    }
    std::fprintf(f, "\n");
}

template<class HostView>
static AETHER_INLINE void dump_plaintext_prims_rows(FILE* f,
                                                    const HostView& prim_h,
                                                    const aether::core::Simulation& sim,
                                                    bool include_ghosts) {
    const auto r = build_index_ranges(sim, include_ghosts);

    if constexpr (AETHER_DIM == 1) {
        for (int i = r.i0; i < r.i1; ++i) {
            write_plain_text_cell_line(f, prim_h, sim, i);
        }
    } else if constexpr (AETHER_DIM == 2) {
        for (int j = r.j0; j < r.j1; ++j) {
            for (int i = r.i0; i < r.i1; ++i) {
                write_plain_text_cell_line(f, prim_h, sim, i, j);
            }
        }
    } else {
        for (int k = r.k0; k < r.k1; ++k) {
            for (int j = r.j0; j < r.j1; ++j) {
                for (int i = r.i0; i < r.i1; ++i) {
                    write_plain_text_cell_line(f, prim_h, sim, i, j, k);
                }
            }
        }
    }
}

static AETHER_INLINE void write_plaintext_snapshot(aether::core::Simulation& sim,
                                                   const std::string& outdir,
                                                   const std::string& prefix,
                                                   bool include_ghosts) {
    const std::string path = make_snapshot_path(outdir, prefix, sim.time.step, ".dat");

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }

    auto prim_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sim.prim);

    write_plaintext_header(f, sim);
    write_plaintext_column_header(f);
    dump_plaintext_prims_rows(f, prim_h, sim, include_ghosts);

    std::fclose(f);
}

void write_snapshot(aether::core::Simulation& sim, snapshot_request& req) {
    for (output_format type : req.formats) {
        switch (type) {
            case output_format::plain_txt:
                write_plaintext_snapshot(sim, req.output_dir, req.prefix, req.include_ghosts);
                break;

            case output_format::binary:
            case output_format::hdf5:
                // not implemented yet
                break;

            default:
                throw std::runtime_error("write_snapshot: unknown output format");
        }
    }
}

} // namespace aether::io