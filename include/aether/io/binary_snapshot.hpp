#pragma once

#include <Kokkos_Core.hpp>

#include <aether/core/config.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/io/metadata.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace aether::io {

constexpr uint64_t AETHER_MAGIC   = 0x4145544845523031ULL; // "AETHER01"
constexpr uint32_t AETHER_VERSION = 1u;

static AETHER_INLINE std::size_t snapshot_scalar_bytes() {
    return sizeof(double);
}

static AETHER_INLINE std::size_t effective_nx(const index_ranges& r) {
    return static_cast<std::size_t>(r.i1 - r.i0);
}

static AETHER_INLINE std::size_t effective_ny(const index_ranges& r) {
    return static_cast<std::size_t>(r.j1 - r.j0);
}

static AETHER_INLINE std::size_t effective_nz(const index_ranges& r) {
    return static_cast<std::size_t>(r.k1 - r.k0);
}

static AETHER_INLINE std::size_t cells_per_var(const index_ranges& r) {
    return effective_nx(r) * effective_ny(r) * effective_nz(r);
}

static AETHER_INLINE std::size_t payload_bytes(const index_ranges& r) {
    return static_cast<std::size_t>(aether::core::Simulation::numvar_full)
         * cells_per_var(r)
         * snapshot_scalar_bytes();
}

static AETHER_INLINE std::size_t binary_offset_of_var(const index_ranges& r, int c) {
    return sizeof(SnapshotHeader)
         + static_cast<std::size_t>(c) * cells_per_var(r) * snapshot_scalar_bytes();
}

static AETHER_INLINE std::size_t binary_offset_of_row(const index_ranges& r,
                                                      int c,
                                                      int k_local,
                                                      int j_local) {
    const std::size_t nx = effective_nx(r);
    const std::size_t ny = effective_ny(r);
    const std::size_t row_index =
        (static_cast<std::size_t>(c) * effective_nz(r) + static_cast<std::size_t>(k_local)) * ny
        + static_cast<std::size_t>(j_local);

    return sizeof(SnapshotHeader) + row_index * nx * snapshot_scalar_bytes();
}

static AETHER_INLINE std::size_t binary_offset_of_1d_chunk(const index_ranges& r,
                                                           int c,
                                                           int i_local) {
    const std::size_t nx = effective_nx(r);
    return sizeof(SnapshotHeader)
         + (static_cast<std::size_t>(c) * nx + static_cast<std::size_t>(i_local))
         * snapshot_scalar_bytes();
}

struct binary_write_task {
    int c{0};

    // logical subregion owned by this task
    int i0{0}, i1{0};   // local [0,nx_eff)
    int j0{0}, j1{0};   // local [0,ny_eff)
    int k0{0}, k1{0};   // local [0,nz_eff)

    std::size_t file_offset{0};
};

template<class HostView>
static AETHER_INLINE void pack_task_buffer(std::vector<double>& buffer,
                                           const HostView& prim_h,
                                           const aether::core::Simulation& sim,
                                           const index_ranges& r,
                                           const binary_write_task& task) {
    const int ng = sim.grid.ng;

    const std::size_t nx_task = static_cast<std::size_t>(task.i1 - task.i0);
    const std::size_t ny_task = static_cast<std::size_t>(task.j1 - task.j0);
    const std::size_t nz_task = static_cast<std::size_t>(task.k1 - task.k0);

    buffer.resize(nx_task * ny_task * nz_task);

    std::size_t p = 0;
    for (int k_local = task.k0; k_local < task.k1; ++k_local) {
        const int k = r.k0 + k_local;
        const int kk = (AETHER_DIM > 2) ? (k + ng) : 0;

        for (int j_local = task.j0; j_local < task.j1; ++j_local) {
            const int j = r.j0 + j_local;
            const int jj = (AETHER_DIM > 1) ? (j + ng) : 0;

            for (int i_local = task.i0; i_local < task.i1; ++i_local) {
                const int i = r.i0 + i_local;
                const int ii = i + ng;
                buffer[p++] = prim_h(task.c, kk, jj, ii);
            }
        }
    }
}

#if !defined(_WIN32)
static AETHER_INLINE void write_task_pwrite(int fd,
                                            const void* data,
                                            std::size_t nbytes,
                                            std::size_t offset) {
    const char* ptr = static_cast<const char*>(data);
    std::size_t written = 0;

    while (written < nbytes) {
        const ssize_t rc = ::pwrite(fd,
                                    ptr + written,
                                    nbytes - written,
                                    static_cast<off_t>(offset + written));
        if (rc < 0) {
            throw std::runtime_error("binary snapshot pwrite failed");
        }
        written += static_cast<std::size_t>(rc);
    }
}
#endif

static AETHER_INLINE std::vector<binary_write_task>
build_binary_tasks(const aether::core::Simulation& sim,
                   const snapshot_request& req) {
    const index_ranges r = build_index_ranges(sim, req.include_ghosts);

    const int nvar = aether::core::Simulation::numvar_full;
    const int nx   = static_cast<int>(effective_nx(r));
    const int ny   = static_cast<int>(effective_ny(r));
    const int nz   = static_cast<int>(effective_nz(r));

    std::vector<binary_write_task> tasks;

    if constexpr (AETHER_DIM == 1) {
        const int target_chunks_per_var = 8;
        const int chunk = (nx + target_chunks_per_var - 1) / target_chunks_per_var;

        for (int c = 0; c < nvar; ++c) {
            for (int i0 = 0; i0 < nx; i0 += chunk) {
                const int i1 = (i0 + chunk < nx) ? (i0 + chunk) : nx;
                binary_write_task t;
                t.c = c;
                t.i0 = i0; t.i1 = i1;
                t.j0 = 0;  t.j1 = 1;
                t.k0 = 0;  t.k1 = 1;
                t.file_offset = binary_offset_of_1d_chunk(r, c, i0);
                tasks.push_back(t);
            }
        }
    } else if constexpr (AETHER_DIM == 2) {
        for (int c = 0; c < nvar; ++c) {
            for (int j = 0; j < ny; ++j) {
                binary_write_task t;
                t.c = c;
                t.i0 = 0;  t.i1 = nx;
                t.j0 = j;  t.j1 = j + 1;
                t.k0 = 0;  t.k1 = 1;
                t.file_offset = binary_offset_of_row(r, c, 0, j);
                tasks.push_back(t);
            }
        }
    } else {
        for (int c = 0; c < nvar; ++c) {
            for (int k = 0; k < nz; ++k) {
                binary_write_task t;
                t.c = c;
                t.i0 = 0;  t.i1 = nx;
                t.j0 = 0;  t.j1 = ny;
                t.k0 = k;  t.k1 = k + 1;
                t.file_offset = binary_offset_of_row(r, c, k, 0);
                tasks.push_back(t);
            }
        }
    }

    return tasks;
}

template<class HostView>
static AETHER_INLINE void write_binary_payload_parallel(const std::string& path,
                                                        const HostView& prim_h,
                                                        const aether::core::Simulation& sim,
                                                        const snapshot_request& req,
                                                        const std::vector<binary_write_task>& tasks) {
#if !defined(_WIN32)
    const int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open binary snapshot for payload writes: " + path);
    }

    using host_exec = Kokkos::DefaultHostExecutionSpace;

    Kokkos::parallel_for(
        "aether_binary_snapshot_write",
        Kokkos::RangePolicy<host_exec>(0, static_cast<int>(tasks.size())),
        [&](const int it) {
            std::vector<double> buffer;
            const index_ranges r = build_index_ranges(sim, req.include_ghosts);
            pack_task_buffer(buffer, prim_h, sim, r, tasks[it]);

            write_task_pwrite(fd,
                              buffer.data(),
                              buffer.size() * sizeof(double),
                              tasks[it].file_offset);
        }
    );
    host_exec().fence();

    ::close(fd);
#else
    // Conservative serial fallback for non-POSIX builds.
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) {
        throw std::runtime_error("Failed to open binary snapshot for payload writes: " + path);
    }

    const index_ranges r = build_index_ranges(sim, req.include_ghosts);
    std::vector<double> buffer;

    for (const auto& task : tasks) {
        pack_task_buffer(buffer, prim_h, sim, r, task);

        if (std::fseek(f, static_cast<long>(task.file_offset), SEEK_SET) != 0) {
            std::fclose(f);
            throw std::runtime_error("binary snapshot fseek failed");
        }

        const std::size_t nw = std::fwrite(buffer.data(), sizeof(double), buffer.size(), f);
        if (nw != buffer.size()) {
            std::fclose(f);
            throw std::runtime_error("binary snapshot fwrite failed");
        }
    }

    std::fclose(f);
#endif
}

static AETHER_INLINE void write_binary_snapshot(aether::core::Simulation& sim,
                                                const snapshot_request& req) {
    const std::string path = make_snapshot_path(req.output_dir, req.prefix, sim.time.write_num, ".bin");
    const index_ranges r   = build_index_ranges(sim, req.include_ghosts);

    auto prim_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sim.view().prim);

    SnapshotHeader hdr{};
    hdr.magic         = AETHER_MAGIC;
    hdr.version       = AETHER_VERSION;
    hdr.step          = static_cast<uint32_t>(sim.time.step);
    hdr.time          = sim.time.t;
    hdr.payload_bytes = payload_bytes(r);

    {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            throw std::runtime_error("Failed to open " + path);
        }

        const std::size_t nw = std::fwrite(&hdr, sizeof(SnapshotHeader), 1, f);
        if (nw != 1) {
            std::fclose(f);
            throw std::runtime_error("Failed to write binary snapshot header to " + path);
        }

        const std::size_t total_bytes = sizeof(SnapshotHeader) + hdr.payload_bytes;
        if (total_bytes > sizeof(SnapshotHeader)) {
            const unsigned char zero = 0;
            if (std::fseek(f, static_cast<long>(total_bytes - 1), SEEK_SET) != 0) {
                std::fclose(f);
                throw std::runtime_error("Failed to size binary snapshot file " + path);
            }
            if (std::fwrite(&zero, 1, 1, f) != 1) {
                std::fclose(f);
                throw std::runtime_error("Failed to finalize size of binary snapshot file " + path);
            }
        }

        std::fclose(f);
    }

    const auto tasks = build_binary_tasks(sim, req);
    write_binary_payload_parallel(path, prim_h, sim, req, tasks);
}

} // namespace aether::io