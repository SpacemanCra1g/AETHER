#pragma once
#include "aether/core/config.hpp"
#include "aether/core/views.hpp"               // Quadrature SoA, flux and cell views
#include <aether/core/RunParams.hpp>           // config struct
#include <aether/core/strides.hpp>             // Extents struct
#include <aether/physics/counts.hpp>           // number of variables
#include <aether/core/config_build.hpp>

namespace aether::core{

    // Simulation struct, templated on the number of Dims
    template <int DIM> struct SimulationD;

    // 1 Dimensional template for Simulation struct
    template <> 
    struct SimulationD<1>{
        // ---------- Set template names to be compile time standard ----------
        using CellsView = CellsViewT<aether::phys_ct::numvar>;
        using CellsSoA = CellsSoAT<aether::phys_ct::numvar>;
        using FaceArrayView = FaceArrayViewT<aether::phys_ct::numvar>;
        using FaceArraySoA = FaceArraySoAT<aether::phys_ct::numvar>;

        // ---------- Sub-structs, containing Time, Grid config, -----------------
        // ---------- and a snapshot only 'view' object, passed by value ----------
        
        struct Time{
            double dt{0.0}, t_start{0.0}, t_end{0.0};
            double t{0.0}, cfl{0.0};
            int step{0};
        };
        struct Grid{
            double dx{0.0}, dy{0.0}, dz{0.0};
            int nx{0}, ny{0}, nz{0};
            double x_min{0.0}, x_max{0.0};
            double y_min{0.0}, y_max{0.0};
            double z_min{0.0}, z_max{0.0};
            int quad{0}, ng{0};
        };
        struct View{
            double dx, dt, cfl,t;
            int nx, quad, ng;
            FaceArrayView x_flux_left;
            FaceArrayView x_flux_right;
            CellsView prim, cons, chars;
        };

        // Compile time known numvar parameter 
        static constexpr int numvar = aether::phys_ct::numvar;

        // Declaring sub-structs
        Config cfg;
        Time time;
        Grid grid;
        // Declaring domain containers
        CellsSoA prims_container;
        CellsSoA cons_container;
        CellsSoA chars_container;
        // Flux and grid extents structs
        Extents ext;
        FaceGridX flux_x_ext;
        // self-explanitory 
        Quadrature quad;
        // Left and Right flux points containers, 
        // each is resized to account for quad points
        FaceArraySoA flux_left_x_container;
        FaceArraySoA flux_right_x_container;
        
        SimulationD() = default;

        // Constructor builds the simulation struct
        explicit SimulationD(const Config &config)
        : cfg(config)
        , time(fill_time(config))
        , grid(fill_grid(config))
        , prims_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , cons_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , chars_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , ext(grid.nx,grid.ny,grid.nz,grid.ng)
        , flux_x_ext(ext), quad(grid.quad)
        , flux_left_x_container(flux_x_ext, quad)
        , flux_right_x_container(flux_x_ext, quad)
        {}
        

        // Returns a View snapshot by value, (except for the domain pointers obv)
        [[nodiscard]] AETHER_INLINE View view() noexcept{
            return
                { grid.dx, time.dt, time.cfl
                , time.t, grid.nx, grid.quad
                , grid.ng, flux_left_x_container.view()
                , flux_right_x_container.view()
                , prims_container.view()
                , cons_container.view()
                , chars_container.view()
            };
            
        }
    // private constructors for the grid and time structs, kept separate to make these 
    // stucts into simple POD types (smaller memory footprint), easy to copy
    private: 
        static AETHER_INLINE Grid fill_grid(const Config &config) noexcept{
            Grid g;
            g.dx = (config.x_end - config.x_start)/config.x_count;
            g.dy = 0.0;
            g.dz = 0.0;

            g.z_min = 0.0; g.z_max = 0.0;
            g.y_min = 0.0; g.y_max = 0.0;
            g.x_min = config.x_start; g.x_max = config.x_end;

            g.ny = 0;
            g.nz = 0;
            g.nx = config.x_count;
            
            g.quad = config.num_quad;
            g.ng = config.num_ghost;
            return g;
        }

        static AETHER_INLINE Time fill_time(const Config &config) noexcept{
            Time t;
            t.dt = 0.0;
            t.t_start = config.t_start;
            t.t = config.t_start;
            t.t_end = config.t_end;
            t.cfl = config.cfl;
            t.step = 0;
            return t;
        }

      
    };

    // 2 Dimensional template for Simulation struct
    template <> 
    struct SimulationD<2>{
        // ---------- Set template names to be compile time standard ----------
        using CellsView = CellsViewT<aether::phys_ct::numvar>;
        using CellsSoA = CellsSoAT<aether::phys_ct::numvar>;
        using FaceArrayView = FaceArrayViewT<aether::phys_ct::numvar>;
        using FaceArraySoA = FaceArraySoAT<aether::phys_ct::numvar>;

        // ---------- Sub-structs, containing Time, Grid config, -----------------
        // ---------- and a snapshot only 'view' object, passed by value ----------
        struct Time{
            double dt{0.0}, t_start{0.0}, t_end{0.0};
            double t{0.0}, cfl{0.0};
            int step{0};
        };
        struct Grid{
            double dx{0.0}, dy{0.0}, dz{0.0};
            int nx{0}, ny{0}, nz{0};
            double x_min{0.0}, x_max{0.0};
            double y_min{0.0}, y_max{0.0};
            double z_min{0.0}, z_max{0.0};
            int quad{0}, ng{0};
        };
        struct View{
            double dx, dy, dt, cfl, t;
            int nx, ny, quad, ng;
            FaceArrayView x_flux_left;
            FaceArrayView x_flux_right;

            FaceArrayView y_flux_left;
            FaceArrayView y_flux_right;

            CellsView prim, cons, chars;
        };

        // Compile time known numvar parameter 
        static constexpr int numvar = aether::phys_ct::numvar;

        // Declaring sub-structs
        Config cfg;
        Time time;
        Grid grid;
        // Declaring domain containers
        CellsSoA prims_container;
        CellsSoA cons_container;
        CellsSoA chars_container;
        // Flux and grid extents structs
        Extents ext;
        FaceGridX flux_x_ext;
        FaceGridY flux_y_ext;
        // self-explanitory 
        Quadrature quad;
        // Left and Right flux points containers, 
        // each is resized to account for quad points
        FaceArraySoA flux_left_x_container;
        FaceArraySoA flux_right_x_container;

        FaceArraySoA flux_left_y_container;
        FaceArraySoA flux_right_y_container;
        
        SimulationD() = default;

        // Constructor builds the simulation struct
        explicit SimulationD(const Config &config)
        : cfg(config)
        , time(fill_time(config))
        , grid(fill_grid(config))
        , prims_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , cons_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , chars_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , ext(grid.nx,grid.ny,grid.nz,grid.ng)
        , flux_x_ext(ext), flux_y_ext(ext)
        , quad(grid.quad)
        , flux_left_x_container(flux_x_ext, quad)
        , flux_right_x_container(flux_x_ext, quad)
        , flux_left_y_container(flux_y_ext, quad)
        , flux_right_y_container(flux_y_ext, quad)
        {}
        

        // Returns a View snapshot by value, (except for the domain pointers obv)
        [[nodiscard]] AETHER_INLINE View view() noexcept{
            return
                { grid.dx, grid.dy, time.dt, time.cfl
                , time.t, grid.nx, grid.ny, grid.quad
                , grid.ng, flux_left_x_container.view()
                , flux_right_x_container.view()
                , flux_left_y_container.view()
                , flux_right_y_container.view()
                , prims_container.view()
                , cons_container.view()
                , chars_container.view()
            };
            
        }
    // private constructors for the grid and time structs, kept separate to make these 
    // stucts into simple POD types (smaller memory footprint), easy to copy
    private: 
        static AETHER_INLINE Grid fill_grid(const Config &config) noexcept{
            Grid g;
            g.dx = (config.x_end - config.x_start)/config.x_count;
            g.dy = (config.y_end - config.y_start)/config.y_count;
            g.dz = 0.0;

            g.z_min = 0.0; g.z_max = 0.0;
            g.y_min = config.y_start; g.y_max = config.y_end;
            g.x_min = config.x_start; g.x_max = config.x_end;

            g.ny = config.y_count;
            g.nz = 0;
            g.nx = config.x_count;
            
            g.quad = config.num_quad;
            g.ng = config.num_ghost;
            return g;
        }

        static AETHER_INLINE Time fill_time(const Config &config) noexcept{
            Time t;
            t.dt = 0.0;
            t.t_start = config.t_start;
            t.t = config.t_start;
            t.t_end = config.t_end;
            t.cfl = config.cfl;
            t.step = 0;
            return t;
        }

      
    };

    template <> 
    struct SimulationD<3>{
        // ---------- Set template names to be compile time standard ----------
        using CellsView = CellsViewT<aether::phys_ct::numvar>;
        using CellsSoA = CellsSoAT<aether::phys_ct::numvar>;
        using FaceArrayView = FaceArrayViewT<aether::phys_ct::numvar>;
        using FaceArraySoA = FaceArraySoAT<aether::phys_ct::numvar>;

        // ---------- Sub-structs, containing Time, Grid config, -----------------
        // ---------- and a snapshot only 'view' object, passed by value ----------
        struct Time{
            double dt{0.0}, t_start{0.0}, t_end{0.0};
            double t{0.0}, cfl{0.0};
            int step{0};
        };
        struct Grid{
            double dx{0.0}, dy{0.0}, dz{0.0};
            int nx{0}, ny{0}, nz{0};
            double x_min{0.0}, x_max{0.0};
            double y_min{0.0}, y_max{0.0};
            double z_min{0.0}, z_max{0.0};
            int quad{0}, ng{0};
        };
        struct View{
            double dx, dy, dz, dt, cfl, t;
            int nx, ny, nz, quad, ng;
            FaceArrayView x_flux_left;
            FaceArrayView x_flux_right;

            FaceArrayView y_flux_left;
            FaceArrayView y_flux_right;

            FaceArrayView z_flux_left;
            FaceArrayView z_flux_right;

            CellsView prim, cons, chars;
        };

        // Compile time known numvar parameter 
        static constexpr int numvar = aether::phys_ct::numvar;

        // Declaring sub-structs
        Config cfg;
        Time time;
        Grid grid;
        // Declaring domain containers
        CellsSoA prims_container;
        CellsSoA cons_container;
        CellsSoA chars_container;
        // Flux and grid extents structs
        Extents ext;
        FaceGridX flux_x_ext;
        FaceGridY flux_y_ext;
        FaceGridY flux_z_ext;
        // self-explanitory 
        Quadrature quad;
        // Left and Right flux points containers, 
        // each is resized to account for quad points
        FaceArraySoA flux_left_x_container;
        FaceArraySoA flux_right_x_container;

        FaceArraySoA flux_left_y_container;
        FaceArraySoA flux_right_y_container;

        FaceArraySoA flux_left_z_container;
        FaceArraySoA flux_right_z_container;
        
        SimulationD() = default;

        // Constructor builds the simulation struct
        explicit SimulationD(const Config &config)
        : cfg(config)
        , time(fill_time(config))
        , grid(fill_grid(config))
        , prims_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , cons_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , chars_container(grid.nx,grid.ny,grid.nz,grid.ng)
        , ext(grid.nx,grid.ny,grid.nz,grid.ng)
        , flux_x_ext(ext), flux_y_ext(ext), flux_z_ext(ext)
        , quad(grid.quad)
        , flux_left_x_container(flux_x_ext, quad)
        , flux_right_x_container(flux_x_ext, quad)
        , flux_left_y_container(flux_y_ext, quad)
        , flux_right_y_container(flux_y_ext, quad)
        , flux_left_z_container(flux_z_ext, quad)
        , flux_right_z_container(flux_z_ext, quad)
        {}
        

        // Returns a View snapshot by value, (except for the domain pointers obv)
        [[nodiscard]] AETHER_INLINE View view() noexcept{
            return
                { grid.dx, grid.dy, grid.dz, time.dt, time.cfl
                , time.t, grid.nx, grid.ny, grid.nz, grid.quad
                , grid.ng, flux_left_x_container.view()
                , flux_right_x_container.view()
                , flux_left_y_container.view()
                , flux_right_y_container.view()
                , flux_left_z_container.view()
                , flux_right_z_container.view()
                , prims_container.view()
                , cons_container.view()
                , chars_container.view()
            };
            
        }
    // private constructors for the grid and time structs, kept separate to make these 
    // stucts into simple POD types (smaller memory footprint), easy to copy
    private: 
        static AETHER_INLINE Grid fill_grid(const Config &config) noexcept{
            Grid g;
            g.dx = (config.x_end - config.x_start)/config.x_count;
            g.dy = (config.y_end - config.y_start)/config.y_count;
            g.dz = (config.z_end - config.z_start)/config.z_count;;

            g.z_min = config.z_start; g.z_max = config.z_end;
            g.y_min = config.y_start; g.y_max = config.y_end;
            g.x_min = config.x_start; g.x_max = config.x_end;

            g.ny = config.y_count;
            g.nz = config.z_count;
            g.nx = config.x_count;
            
            g.quad = config.num_quad;
            g.ng = config.num_ghost;
            return g;
        }

        static AETHER_INLINE Time fill_time(const Config &config) noexcept{
            Time t;
            t.dt = 0.0;
            t.t_start = config.t_start;
            t.t = config.t_start;
            t.t_end = config.t_end;
            t.cfl = config.cfl;
            t.step = 0;
            return t;
        }
    };

    using Simulation = SimulationD<AETHER_DIM>;
}