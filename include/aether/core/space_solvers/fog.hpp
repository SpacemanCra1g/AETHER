#pragma once 
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/core/stencil_templates.hpp>


namespace aether::core{

template<int numvar, sweep_dir dir, class FaceViewT>
AETHER_INLINE void FOG_face_from_stencil(const Stencil1D<0, numvar, dir>& S,
                                        FaceViewT& FL,
                                        FaceViewT& FR,                                        
                                        const std::size_t fidL, 
                                        const std::size_t fidR) noexcept
{
    #pragma omp simd
    for (int v = 0; v < numvar; ++v) {
        const double q = S.get(v, 0);
        // Note the swapped indexing here (FR -> fidL) There is no good way to 
        // swap the reference frame (left from cell center or from interface frame of refference)
        // This is where I am choosing to do it I guess 
        FR.comp[v][fidL] = q;
        FL.comp[v][fidR] = q;
    }
}
}