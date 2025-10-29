#include <aether/physics/euler/convert.hpp>
#include <aether/core/prim_layout.hpp>
#include <aether/core/con_layout.hpp>

using P = aether::prim::Prim;
using C = aether::con::Cons;

namespace aether::physics::euler {
void cons_to_prims_domain(aether::core::Simulation::View &view, const double gamma){
    // 1D version 
#if AETHER_DIM == 1
    const int Nx = view.nx + view.ng;
    #pragma omp parallel for schedule(static) default(none) shared(view,gamma,Nx)
    for (int i = -view.ng; i < Nx; ++i){
        cons con; 
        con.rho = view.cons.var(C::RHO,i,0,0); 
        con.mx = view.cons.var(C::MX,i,0,0); 
        con.E = view.cons.var(C::E,i,0,0); 
        auto prim = cons_to_prims_cell(con,gamma);

        view.prim.var(P::RHO,i,0,0) = prim.rho;
        view.prim.var(P::VX,i,0,0) = prim.vx;
        view.prim.var(P::P,i,0,0) = prim.p;
    }

#elif AETHER_DIM == 2
    const int Nx = view.nx + view.ng;
    const int Ny = view.ny + view.ng;
    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(view,gamma,Nx,Ny)
    for (int j = -view.ng; j < Ny; ++j){
        for (int i = -view.ng; i < Nx; ++i){
            cons con; 
            con.rho = view.cons.var(C::RHO,i,j,0); 
            con.mx = view.cons.var(C::MX,i,j,0); 
            con.my = view.cons.var(C::MY,i,j,0); 
            con.E = view.cons.var(C::E,i,j,0); 
            auto prim = cons_to_prims_cell(con,gamma);

            view.prim.var(P::RHO,i,j,0) = prim.rho;
            view.prim.var(P::VX,i,j,0) = prim.vx;
            view.prim.var(P::VY,i,j,0) = prim.vy;
            view.prim.var(P::P,i,j,0) = prim.p;
        }
    }
#elif AETHER_DIM == 3
    const int Nx = view.nx + view.ng;
    const int Ny = view.ny + view.ng;
    const int Nz = view.nz + view.ng;
    #pragma omp parallel for collapse(3) schedule(static) default(none) shared(view,gamma,Nx,Ny,Nz)
    for (int k = -view.ng; k < Nz; ++k){
        for (int j = -view.ng; j < Ny; ++j){
            for (int i = -view.ng; i < Nx; ++i){
                cons con; 
                con.rho = view.cons.var(C::RHO,i,j,k); 
                con.mx = view.cons.var(C::MX,i,j,k); 
                con.my = view.cons.var(C::MY,i,j,k); 
                con.mz = view.cons.var(C::MZ,i,j,k); 
                con.E = view.cons.var(C::E,i,j,k); 
                auto prim = cons_to_prims_cell(con,gamma);

                view.prim.var(P::RHO,i,j,k) = prim.rho;
                view.prim.var(P::VX,i,j,k) = prim.vx;
                view.prim.var(P::VY,i,j,k) = prim.vy;
                view.prim.var(P::VZ,i,j,k) = prim.vz;
                view.prim.var(P::P,i,j,k) = prim.p;
            }
        }
    }   
#endif

}

void prims_to_cons_domain(aether::core::Simulation::View &view, const double gamma){
    // 1D version 
#if AETHER_DIM == 1
    const int Nx = view.nx + view.ng;
    #pragma omp parallel for schedule(static) default(none) shared(view,gamma,Nx)
    for (int i = -view.ng; i < Nx; ++i){
        prims prim; 
        prim.rho = view.prim.var(P::RHO,i,0,0); 
        prim.vx = view.prim.var(P::VX,i,0,0); 
        prim.p = view.prim.var(P::P,i,0,0); 
        auto con = prims_to_cons_cell(prim,gamma);

        view.cons.var(C::RHO,i,0,0) = con.rho;
        view.cons.var(C::MX,i,0,0) =  con.mx;
        view.cons.var(C::E,i,0,0) =   con.E;
    }

#elif AETHER_DIM == 2
    const int Nx = view.nx + view.ng;
    const int Ny = view.ny + view.ng;
    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(view,gamma,Nx,Ny)
    for (int j = -view.ng; j < Ny; ++j){
        for (int i = -view.ng; i < Nx; ++i){
            prims prim; 
            prim.rho = view.prim.var(P::RHO,i,j,0); 
            prim.vx = view.prim.var(P::VX,i,j,0); 
            prim.vy = view.prim.var(P::VY,i,j,0); 
            prim.p = view.prim.var(P::P,i,j,0); 
            auto con = prims_to_cons_cell(prim,gamma);

            view.cons.var(C::RHO,i,j,0) = con.rho;
            view.cons.var(C::MX,i,j,0) =  con.mx;
            view.cons.var(C::MY,i,j,0) =  con.my;
            view.cons.var(C::E,i,j,0) =   con.E;
        }
    }
#elif AETHER_DIM == 3
    const int Nx = view.nx + view.ng;
    const int Ny = view.ny + view.ng;
    const int Nz = view.nz + view.ng;
    #pragma omp parallel for collapse(3) schedule(static) default(none) shared(view,gamma,Nx,Ny,Nz)
    for (int k = -view.ng; k < Nz; ++k){
        for (int j = -view.ng; j < Ny; ++j){
            for (int i = -view.ng; i < Nx; ++i){
                prims prim; 
                prim.rho = view.prim.var(P::RHO,i,j,k); 
                prim.vx = view.prim.var(P::VX,i,j,k); 
                prim.vy = view.prim.var(P::VY,i,j,k); 
                prim.vz = view.prim.var(P::VZ,i,j,k); 
                prim.p = view.prim.var(P::P,i,j,k); 
                auto con = prims_to_cons_cell(prim,gamma);

                view.cons.var(C::RHO,i,j,k) = con.rho;
                view.cons.var(C::MX,i,j,k) =  con.mx;
                view.cons.var(C::MY,i,j,k) =  con.my;
                view.cons.var(C::MZ,i,j,k) =  con.mz;
                view.cons.var(C::E,i,j,k) =   con.E;
            }
        }
    }   
#endif

}
}