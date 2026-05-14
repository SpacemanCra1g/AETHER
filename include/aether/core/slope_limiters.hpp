#pragma once 
#include "Kokkos_Macros.hpp"
#include <aether/math/mats.hpp>
#include <cmath>

namespace aether::core{

KOKKOS_INLINE_FUNCTION 
double sign(double x) {return (x >= 0.0) - (x < 0.0);}

template <int N>
KOKKOS_INLINE_FUNCTION
math::Vec<N> minmod(math::Vec<N> left, math::Vec<N> right){
    math::Vec<N> out;
    double a,b,f_a,f_b;
    for (int i = 0; i < N; ++i){
        a = left[i];
        b = right[i];
        f_a = fabs(a);
        f_b = fabs(b);
        out[i] = (a*b < 0) ? 0.0 : (f_a < f_b) ? a : b;
    }
    return out;
}

// Overloads for the minmod function
KOKKOS_INLINE_FUNCTION
double minmod(double a, double b){
	const double f_a = fabs(a);
	const double f_b = fabs(b);
	return (a*b < 0) ? 0.0 : (f_a < f_b) ? a : b;
}

template <int N>
KOKKOS_INLINE_FUNCTION
math::Vec<N> van_leer(math::Vec<N> left, math::Vec<N> right){
    math::Vec<N> out;
    double a,b;
    for (int i = 0; i < N; ++i){
        a = left[i];
        b = right[i];
        out[i] = (a*b <= 0.0) ? 0.0 : (2.0 * a * b) / (a + b);
    }
    return out;
}

// Overloads for the van_leer function
KOKKOS_INLINE_FUNCTION
double van_leer(double a, double b){
	return (a*b <= 0.0) ? 0.0 : (2.0 * a * b) / (a + b);
}

template <int N>
KOKKOS_INLINE_FUNCTION
math::Vec<N> mc(math::Vec<N> left, math::Vec<N> middle, math::Vec<N> right){
    return minmod(left, minmod(middle,right));
}

KOKKOS_INLINE_FUNCTION
double mc(double left, double middle, double right){
    return minmod(left, minmod(middle,right));
}

}
