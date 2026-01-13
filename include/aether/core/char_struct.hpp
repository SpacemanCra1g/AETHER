#pragma once 
#include "aether/core/config.hpp"
#include <array>
#include <vector>
#include <aether/math/mats.hpp>
#include <aether/physics/counts.hpp>

namespace aether::core {


template <int Dim,int numvar>
struct eigenvec_viewT{
    aether::math::Mat<numvar> *x_left, *x_right, *y_left, *y_right, *z_left,*z_right;
    std::array<double,5> *x_eigs, *y_eigs, *z_eigs;
    bool *populated;

};
using eigenvec_view = eigenvec_viewT<AETHER_DIM, aether::phys_ct::numvar>;


template <int Dim,int numvar>
struct EigenvectorsT {

std::vector<aether::math::Mat<numvar> > x_left, x_right, y_left, y_right, z_left, z_right;
std::vector<std::array<double,5>> x_eigs, y_eigs, z_eigs;
bool populated = false;

EigenvectorsT() = default;

EigenvectorsT(const int num_cells){

    x_left.resize(num_cells);
    x_right.resize(num_cells);
    x_eigs.resize(num_cells);
    if constexpr (Dim > 1) {
    y_left.resize(num_cells);
    y_right.resize(num_cells);
    y_eigs.resize(num_cells);
    }
    if constexpr (Dim > 2) {
    z_left.resize(num_cells);
    z_right.resize(num_cells);
    z_eigs.resize(num_cells);
    }
}

[[nodiscard]] AETHER_INLINE eigenvec_viewT<Dim, numvar> view() noexcept{
    eigenvec_viewT<Dim, numvar> v;

    v.y_left = nullptr;
    v.z_left = nullptr;
    v.y_right = nullptr;
    v.z_right = nullptr;
    v.y_eigs = nullptr;
    v.z_eigs = nullptr;

    v.x_left = x_left.data();
    v.x_right = x_right.data();
    v.x_eigs = x_eigs.data();
    v.populated = &populated;

    if constexpr (Dim > 1) {
    v.y_left = y_left.data();    
    v.y_right = y_right.data();
    v.y_eigs = y_eigs.data();
    }

    if constexpr (Dim > 2) {
    v.z_left = z_left.data();
    v.z_right = z_right.data();
    v.z_eigs = z_eigs.data();
    }
    return v;
}

};

struct one_cell_spectral_container{
    aether::math::Mat<phys_ct::numvar>  *x_left, *x_right,
                                        *y_left, *y_right,
                                        *z_left, *z_right;
    std::array<double,5> *x_eigs, *y_eigs, *z_eigs;
};

using eigenvectors = EigenvectorsT<AETHER_DIM, aether::phys_ct::numvar>;

}
