#pragma once
#include "aether/physics/counts.hpp"
#include <aether/core/views.hpp>
#include <aether/core/char_struct.hpp>
#include <aether/math/mats.hpp>
#include <aether/core/enums.hpp>
#include <cstddef>
#include <stdexcept>

namespace aether::core{
template<int Dim>
AETHER_INLINE void prim_to_charT(const CellsView &prim, CharView &chars, const eigenvec_view &eigs);

template <>
AETHER_INLINE void prim_to_charT<1>(const CellsView &prim, CharView &chars, const eigenvec_view &eigs){
    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N,numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = prim.var(j,i);
        
        auto Ch = eigs.x_left[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) chars.var(0, j, i) = Ch[j];
    }
}

template <>
AETHER_INLINE void prim_to_charT<2>(const CellsView &prim, CharView &chars, const eigenvec_view &eigs){
    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N,numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = prim.var(j,i);
        
        auto Ch = eigs.x_left[i].dot(P);
        auto Ch2 = eigs.y_left[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) {
            chars.var(0, j, i) = Ch[j];
            chars.var(1, j, i) = Ch2[j];
        }
    }
}

template <>
AETHER_INLINE void prim_to_charT<3>(const CellsView &prim, CharView &chars, const eigenvec_view &eigs){
    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N,numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = prim.var(j,i);
        
        auto Ch = eigs.x_left[i].dot(P);
        auto Ch2 = eigs.y_left[i].dot(P);
        auto Ch3 = eigs.z_left[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) {
            chars.var(0, j, i) = Ch[j];
            chars.var(1, j, i) = Ch2[j];
            chars.var(2, j, i) = Ch3[j];
        }
    }
}

template<sweep_dir val>
AETHER_INLINE void char_to_primT(const CharView &chars, CellsView &prim, const eigenvec_view &eigs);

// These are templated on which dimension we are converting back from. 
// In other words, taking the linearization from the x,y,or z dimensions 

// X-direction
template <>
AETHER_INLINE void char_to_primT<sweep_dir::x>(const CharView &chars, CellsView &prim, const eigenvec_view &eigs){
    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N, numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = chars.var(0,j,i);
        
        auto Pr = eigs.x_right[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) prim.var(j, i) = Pr[j];
    }
}

// Y-direction
template <>
AETHER_INLINE void char_to_primT<sweep_dir::y>(const CharView &chars, CellsView &prim, const eigenvec_view &eigs){
    if constexpr (AETHER_DIM < 2) {
        throw std::runtime_error("z-characteristics called in not 3D problem");
    }
    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N, numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = chars.var(1,j,i);
        
        auto Pr = eigs.y_right[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) prim.var(j, i) = Pr[j];
    }
}

template <>
AETHER_INLINE void char_to_primT<sweep_dir::z>(const CharView &chars, CellsView &prim, const eigenvec_view &eigs){
    if constexpr (AETHER_DIM < 3) {
        throw std::runtime_error("z-characteristics called in not 3D problem");
    }

    std::size_t N = prim.ext.flat();
    constexpr int numvar = aether::phys_ct::numvar;

    #pragma omp parallel for schedule(static) default(none) shared(prim,chars,eigs,N, numvar)
    for (std::size_t i = 0; i < N; ++i){

        aether::math::Vec<numvar> P; 
        #pragma omp simd
        for (int j = 0; j < numvar; ++j) P[j] = chars.var(2,j,i);
        
        auto Pr = eigs.z_right[i].dot(P);

        #pragma omp simd
        for (int j = 0; j < numvar; ++j) prim.var(j, i) = Pr[j];
    }
}

constexpr auto prim_to_char = prim_to_charT<AETHER_DIM>;
constexpr auto x_char_to_prim = char_to_primT<sweep_dir::x>;
constexpr auto y_char_to_prim = char_to_primT<sweep_dir::y>;
constexpr auto z_char_to_prim = char_to_primT<sweep_dir::z>;

}

