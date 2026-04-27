#pragma once
#include <aether/core/config.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/io/metadata.hpp>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace aether::io{

static AETHER_INLINE void write_plaintext_header(
    FILE* f,
    const aether::core::Simulation& sim,
    const snapshot_request& req,
    const std::string& snapshot_path,
    const index_ranges& r
) {
    namespace fs = std::filesystem;

    const auto& g = sim.grid;
    const auto& t = sim.time;

    std::fprintf(f, "# AETHER snapshot (PlainText)\n");
    std::fprintf(f, "# metadata_file=%s\n",
                 fs::path(make_metadata_path(req.output_dir, req.prefix)).filename().string().c_str());
    std::fprintf(f, "# snapshot_file=%s\n",
                 fs::path(snapshot_path).filename().string().c_str());

    std::fprintf(f, "# dim=%d  numvar=%d  step=%d\n",
                 AETHER_DIM, aether::core::Simulation::numvar, t.step);
    std::fprintf(f, "# t=%.16e  dt=%.16e  cfl=%.6f\n", t.t, t.dt, t.cfl);

    std::fprintf(f, "# nx=%d", g.nx);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "  ny=%d", g.ny);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "  nz=%d", g.nz);
    std::fprintf(f, "  ng=%d  quad=%d  gamma=%.6f\n",
                 g.ng, g.quad, g.gamma);

    std::fprintf(f, "# include_ghosts=%s\n", req.include_ghosts ? "true" : "false");

    std::fprintf(f, "# write_ranges: i[%d,%d)", r.i0, r.i1);
    if constexpr (AETHER_DIM >= 2) std::fprintf(f, "  j[%d,%d)", r.j0, r.j1);
    if constexpr (AETHER_DIM == 3) std::fprintf(f, "  k[%d,%d)", r.k0, r.k1);
    std::fprintf(f, "\n");

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
        "# Variable order: indices (i[,j[,k]]) then state variables v0..v%d plus contact_wave\n",
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

    std::fprintf(f, "  contact_wave\n");
}

template<class HostView, class ContactView>
static AETHER_INLINE void write_plain_text_cell_line(FILE* f,
                                                     const HostView& prim_h,
                                                     const ContactView& contact_h,
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

    // NEW: contact_wave output
    std::fprintf(f, "  %.16e", contact_h(kk, jj, ii));

    std::fprintf(f, "\n");
}

template<class HostView, class ContactView>
static AETHER_INLINE void dump_plaintext_prims_rows(FILE* f,
                                                    const HostView& prim_h,
                                                    const ContactView& contact_h,
                                                    const aether::core::Simulation& sim,
                                                    bool include_ghosts) {
    const auto r = build_index_ranges(sim, include_ghosts);

    if constexpr (AETHER_DIM == 1) {
        for (int i = r.i0; i < r.i1; ++i) {
            write_plain_text_cell_line(f, prim_h, contact_h, sim, i);
        }
    } else if constexpr (AETHER_DIM == 2) {
        for (int j = r.j0; j < r.j1; ++j) {
            for (int i = r.i0; i < r.i1; ++i) {
                write_plain_text_cell_line(f, prim_h, contact_h, sim, i, j);
            }
        }
    } else {
        for (int k = r.k0; k < r.k1; ++k) {
            for (int j = r.j0; j < r.j1; ++j) {
                for (int i = r.i0; i < r.i1; ++i) {
                    write_plain_text_cell_line(f, prim_h, contact_h, sim, i, j, k);
                }
            }
        }
    }
}

static AETHER_INLINE void write_plaintext_snapshot(aether::core::Simulation& sim,
                                                   const snapshot_request& req) {
    const std::string path = make_snapshot_path(req.output_dir, req.prefix, sim.time.write_num, ".dat");
    const auto r = build_index_ranges(sim, req.include_ghosts);

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }

    auto prim_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sim.view().prim);
    auto contact_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sim.view().contact_wave);

    write_plaintext_header(f, sim, req, path, r);
    write_plaintext_column_header(f);
    dump_plaintext_prims_rows(f, prim_h, contact_h, sim, req.include_ghosts);

    std::fclose(f);
}

}