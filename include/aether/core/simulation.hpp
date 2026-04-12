#pragma once
#include <array>
#include <aether/core/config.hpp>
#include <aether/core/config_build.hpp>
#include <aether/core/Kokkos_Policy.hpp>
#include <aether/core/RunParams.hpp>
#include <aether/core/strides.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/counts.hpp>

namespace aether::core {

template<int DIM>
struct SimulationD;

// ============================================================
// Shared runtime metadata
// ============================================================

static AETHER_INLINE bool compute_ctu_enabled(const Config& c) noexcept {
    return AETHER_DIM > 1 && (c.solve == solver::fog || c.solve == solver::plm );
}

struct TimeState {
    double dt{0.0};
    double t_start{0.0};
    double t_end{0.0};
    double t{0.0};
    double cfl{0.0};
    int step{0};
    int RK_stage{0};
};

struct GridState {
    double dx{0.0}, dy{0.0}, dz{0.0};

    int nx{0}, ny{1}, nz{1};

    double x_min{0.0}, x_max{0.0};
    double y_min{0.0}, y_max{0.0};
    double z_min{0.0}, z_max{0.0};

    int quad{1};
    int ng{0};

    double gamma{0.0};
};

// ============================================================
// 1D
// ============================================================

template<>
struct SimulationD<1> {
    static constexpr int dim = 1;
    static constexpr int numvar = aether::phys_ct::numvar;

    using policy_type = aether::kokkos_cfg::Policy<1, AETHER_PHYSICS_KIND>;

    using CellView = typename policy_type::template CellView<double>;
    using DirView  = typename policy_type::template DirCellView<double>;
    using FaceView = typename policy_type::template FaceView<double>;

    Config cfg{};
    TimeState time{};
    GridState grid{};

    CellGrid<1> cells{};
    FaceGridX xfaces{};

    CellView prim{};
    CellView cons{};
    DirView  chars{};

    FaceView fxL{};
    FaceView fxR{};
    FaceView fx{};
    bool ctu_enabled{false};

    std::array<sweep_dir, 1> sweeps{ sweep_dir::x };

    struct View {
        double dx, dy, dz;
        double dt, cfl, t;
        int nx, ny, nz;
        int ng, quad;
        double gamma;

        CellGrid<1> cells;
        FaceGridX   xfaces;

        CellView prim;
        CellView cons;
        DirView  chars;

        FaceView fxL;
        FaceView fxR;
        FaceView fx;
    };

    SimulationD() = default;

    explicit SimulationD(const Config& config)
        : cfg(config),
          time(fill_time(config)),
          grid(fill_grid(config)),
          cells(grid.nx, grid.ng),
          xfaces(cells),
          prim("prim", numvar, cells.Nz, cells.Ny, cells.Nx),
          cons("cons", numvar, cells.Nz, cells.Ny, cells.Nx),
          chars("chars", dim, numvar, cells.Nz, cells.Ny, cells.Nx),
          fxL("fxL", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fxR("fxR", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fx ("fx",  numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          ctu_enabled(compute_ctu_enabled(config))
    {}

    [[nodiscard]] AETHER_INLINE
    View view() const noexcept {
        return {
            grid.dx, grid.dy, grid.dz,
            time.dt, time.cfl, time.t,
            grid.nx, grid.ny, grid.nz,
            grid.ng, grid.quad,
            grid.gamma,
            cells,
            xfaces,
            prim,
            cons,
            chars,
            fxL,
            fxR,
            fx
        };
    }

private:
    static AETHER_INLINE GridState fill_grid(const Config& c) noexcept {
        GridState g;
        g.dx = (c.x_end - c.x_start) / c.x_count;
        g.dy = 0.0;
        g.dz = 0.0;

        g.nx = c.x_count;
        g.ny = 1;
        g.nz = 1;

        g.x_min = c.x_start; g.x_max = c.x_end;
        g.y_min = 0.0;       g.y_max = 0.0;
        g.z_min = 0.0;       g.z_max = 0.0;

        g.quad  = c.num_quad;
        g.ng    = c.num_ghost;
        g.gamma = c.gamma;
        return g;
    }

    static AETHER_INLINE TimeState fill_time(const Config& c) noexcept {
        TimeState t;
        t.dt       = 0.0;
        t.t_start  = c.t_start;
        t.t_end    = c.t_end;
        t.t        = c.t_start;
        t.cfl      = c.cfl;
        t.step     = 0;
        t.RK_stage = 0;
        return t;
    }
};

// ============================================================
// 2D
// ============================================================

template<>
struct SimulationD<2> {
    static constexpr int dim = 2;
    static constexpr int numvar = aether::phys_ct::numvar;

    using policy_type = aether::kokkos_cfg::Policy<2, AETHER_PHYSICS_KIND>;

    using CellView = typename policy_type::template CellView<double>;
    using DirView  = typename policy_type::template DirCellView<double>;
    using FaceView = typename policy_type::template FaceView<double>;

    Config cfg{};
    TimeState time{};
    GridState grid{};

    CellGrid<2> cells{};
    FaceGridX xfaces{};
    FaceGridY yfaces{};

    CellView prim{};
    CellView cons{};
    DirView  chars{};

    FaceView fxL{};
    FaceView fxR{};
    FaceView fx{};

    FaceView fyL{};
    FaceView fyR{};
    FaceView fy{};
    bool ctu_enabled{false};

    std::array<sweep_dir, 2> sweeps{ sweep_dir::x, sweep_dir::y };

    struct View {
        double dx, dy, dz;
        double dt, cfl, t;
        int nx, ny, nz;
        int ng, quad;
        double gamma;

        CellGrid<2> cells;
        FaceGridX   xfaces;
        FaceGridY   yfaces;

        CellView prim;
        CellView cons;
        DirView  chars;

        FaceView fxL;
        FaceView fxR;
        FaceView fx;

        FaceView fyL;
        FaceView fyR;
        FaceView fy;
    };

    SimulationD() = default;

    explicit SimulationD(const Config& config)
        : cfg(config),
          time(fill_time(config)),
          grid(fill_grid(config)),
          cells(grid.nx, grid.ny, grid.ng),
          xfaces(cells),
          yfaces(cells),
          prim("prim", numvar, cells.Nz, cells.Ny, cells.Nx),
          cons("cons", numvar, cells.Nz, cells.Ny, cells.Nx),
          chars("chars", dim, numvar, cells.Nz, cells.Ny, cells.Nx),
          fxL("fxL", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fxR("fxR", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fx ("fx",  numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fyL("fyL", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          fyR("fyR", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          fy ("fy",  numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          ctu_enabled(compute_ctu_enabled(config))
    {}

    [[nodiscard]] AETHER_INLINE
    View view() const noexcept {
        return {
            grid.dx, grid.dy, grid.dz,
            time.dt, time.cfl, time.t,
            grid.nx, grid.ny, grid.nz,
            grid.ng, grid.quad,
            grid.gamma,
            cells,
            xfaces,
            yfaces,
            prim,
            cons,
            chars,
            fxL,
            fxR,
            fx,
            fyL,
            fyR,
            fy
        };
    }

private:
    static AETHER_INLINE GridState fill_grid(const Config& c) noexcept {
        GridState g;
        g.dx = (c.x_end - c.x_start) / c.x_count;
        g.dy = (c.y_end - c.y_start) / c.y_count;
        g.dz = 0.0;

        g.nx = c.x_count;
        g.ny = c.y_count;
        g.nz = 1;

        g.x_min = c.x_start; g.x_max = c.x_end;
        g.y_min = c.y_start; g.y_max = c.y_end;
        g.z_min = 0.0;       g.z_max = 0.0;

        g.quad  = c.num_quad;
        g.ng    = c.num_ghost;
        g.gamma = c.gamma;
        return g;
    }

    static AETHER_INLINE TimeState fill_time(const Config& c) noexcept {
        TimeState t;
        t.dt       = 0.0;
        t.t_start  = c.t_start;
        t.t_end    = c.t_end;
        t.t        = c.t_start;
        t.cfl      = c.cfl;
        t.step     = 0;
        t.RK_stage = 0;
        return t;
    }
};

// ============================================================
// 3D
// ============================================================

template<>
struct SimulationD<3> {
    static constexpr int dim = 3;
    static constexpr int numvar = aether::phys_ct::numvar;

    using policy_type = aether::kokkos_cfg::Policy<3, AETHER_PHYSICS_KIND>;

    using CellView = typename policy_type::template CellView<double>;
    using DirView  = typename policy_type::template DirCellView<double>;
    using FaceView = typename policy_type::template FaceView<double>;

    Config cfg{};
    TimeState time{};
    GridState grid{};

    CellGrid<3> cells{};
    FaceGridX xfaces{};
    FaceGridY yfaces{};
    FaceGridZ zfaces{};

    CellView prim{};
    CellView cons{};
    DirView  chars{};

    FaceView fxL{};
    FaceView fxR{};
    FaceView fx{};

    FaceView fyL{};
    FaceView fyR{};
    FaceView fy{};

    FaceView fzL{};
    FaceView fzR{};
    FaceView fz{};

    bool ctu_enabled{false};

    FaceView ctu_fxL{};
    FaceView ctu_fxR{};
    FaceView ctu_fx{};

    FaceView ctu_fyL{};
    FaceView ctu_fyR{};
    FaceView ctu_fy{};

    FaceView ctu_fzL{};
    FaceView ctu_fzR{};
    FaceView ctu_fz{};

    FaceView ctu_xL_bak{};
    FaceView ctu_xR_bak{};
    FaceView ctu_yL_bak{};
    FaceView ctu_yR_bak{};
    FaceView ctu_zL_bak{};
    FaceView ctu_zR_bak{};

    std::array<sweep_dir, 3> sweeps{ sweep_dir::x, sweep_dir::y, sweep_dir::z };

    struct View {
        double dx, dy, dz;
        double dt, cfl, t;
        int nx, ny, nz;
        int ng, quad;
        double gamma;

        CellGrid<3> cells;
        FaceGridX   xfaces;
        FaceGridY   yfaces;
        FaceGridZ   zfaces;

        CellView prim;
        CellView cons;
        DirView  chars;

        FaceView fxL;
        FaceView fxR;
        FaceView fx;

        FaceView fyL;
        FaceView fyR;
        FaceView fy;

        FaceView fzL;
        FaceView fzR;
        FaceView fz;
    };

    struct CTUView {
        bool enabled{false};

        FaceView fxL;
        FaceView fxR;
        FaceView fx;

        FaceView fyL;
        FaceView fyR;
        FaceView fy;

        FaceView fzL;
        FaceView fzR;
        FaceView fz;

        FaceView xL_bak;
        FaceView xR_bak;
        FaceView yL_bak;
        FaceView yR_bak;
        FaceView zL_bak;
        FaceView zR_bak;
    };

    SimulationD() = default;

    explicit SimulationD(const Config& config)
        : cfg(config),
          time(fill_time(config)),
          grid(fill_grid(config)),
          cells(grid.nx, grid.ny, grid.nz, grid.ng),
          xfaces(cells),
          yfaces(cells),
          zfaces(cells),
          prim("prim", numvar, cells.Nz, cells.Ny, cells.Nx),
          cons("cons", numvar, cells.Nz, cells.Ny, cells.Nx),
          chars("chars", dim, numvar, cells.Nz, cells.Ny, cells.Nx),
          fxL("fxL", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fxR("fxR", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fx ("fx",  numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx),
          fyL("fyL", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          fyR("fyR", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          fy ("fy",  numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx),
          fzL("fzL", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx),
          fzR("fzR", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx),
          fz ("fz",  numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx),
          ctu_enabled(compute_ctu_enabled(config))
    {
        if (ctu_enabled) {
            ctu_fxL = FaceView("ctu_fxL", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx);
            ctu_fxR = FaceView("ctu_fxR", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx);
            ctu_fx  = FaceView("ctu_fx",  numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx);

            ctu_fyL = FaceView("ctu_fyL", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx);
            ctu_fyR = FaceView("ctu_fyR", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx);
            ctu_fy  = FaceView("ctu_fy",  numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx);

            ctu_fzL = FaceView("ctu_fzL", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx);
            ctu_fzR = FaceView("ctu_fzR", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx);
            ctu_fz  = FaceView("ctu_fz",  numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx);

            ctu_xL_bak = FaceView("ctu_xL_bak", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx);
            ctu_xR_bak = FaceView("ctu_xR_bak", numvar, grid.quad, xfaces.Nz, xfaces.Ny, xfaces.Nfx);

            ctu_yL_bak = FaceView("ctu_yL_bak", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx);
            ctu_yR_bak = FaceView("ctu_yR_bak", numvar, grid.quad, yfaces.Nz, yfaces.Nfy, yfaces.Nx);

            ctu_zL_bak = FaceView("ctu_zL_bak", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx);
            ctu_zR_bak = FaceView("ctu_zR_bak", numvar, grid.quad, zfaces.Nfz, zfaces.Ny, zfaces.Nx);
        }
    }

    [[nodiscard]] AETHER_INLINE
    View view() const noexcept {
        return {
            grid.dx, grid.dy, grid.dz,
            time.dt, time.cfl, time.t,
            grid.nx, grid.ny, grid.nz,
            grid.ng, grid.quad,
            grid.gamma,
            cells,
            xfaces,
            yfaces,
            zfaces,
            prim,
            cons,
            chars,
            fxL,
            fxR,
            fx,
            fyL,
            fyR,
            fy,
            fzL,
            fzR,
            fz
        };
    }

    [[nodiscard]] AETHER_INLINE
    CTUView ctu_view() const noexcept {
        return {
            ctu_enabled,
            ctu_fxL, ctu_fxR, ctu_fx,
            ctu_fyL, ctu_fyR, ctu_fy,
            ctu_fzL, ctu_fzR, ctu_fz,
            ctu_xL_bak, ctu_xR_bak,
            ctu_yL_bak, ctu_yR_bak,
            ctu_zL_bak, ctu_zR_bak
        };
    }

    [[nodiscard]] AETHER_INLINE
    bool ctu_active() const noexcept {
        return ctu_enabled;
    }

private:
    static AETHER_INLINE GridState fill_grid(const Config& c) noexcept {
        GridState g;
        g.dx = (c.x_end - c.x_start) / c.x_count;
        g.dy = (c.y_end - c.y_start) / c.y_count;
        g.dz = (c.z_end - c.z_start) / c.z_count;

        g.nx = c.x_count;
        g.ny = c.y_count;
        g.nz = c.z_count;

        g.x_min = c.x_start; g.x_max = c.x_end;
        g.y_min = c.y_start; g.y_max = c.y_end;
        g.z_min = c.z_start; g.z_max = c.z_end;

        g.quad  = c.num_quad;
        g.ng    = c.num_ghost;
        g.gamma = c.gamma;
        return g;
    }

    static AETHER_INLINE TimeState fill_time(const Config& c) noexcept {
        TimeState t;
        t.dt       = 0.0;
        t.t_start  = c.t_start;
        t.t_end    = c.t_end;
        t.t        = c.t_start;
        t.cfl      = c.cfl;
        t.step     = 0;
        t.RK_stage = 0;
        return t;
    }


};

// ============================================================
// Active build alias
// ============================================================

using Simulation = SimulationD<AETHER_DIM>;
using CellView      = Simulation::CellView;
using DirView       = Simulation::DirView;
using FaceView      = Simulation::FaceView;

} // namespace aether::core