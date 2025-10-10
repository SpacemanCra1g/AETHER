#pragma once 
#include <array>
#include <vector>
#include <cstddef>
#include <cassert>
#include <aether/core/config.hpp>
#include <aether/core/strides.hpp>


namespace aether::core {

// ---------- Start with a generic Cell Centered array structure ----------
// ---------- Template on the number of components (compile time known, physics & dim) ----------
template<int NCOMP> 
struct CellsViewT{
    std::array<double*, NCOMP> comp{};  // comp[NCOMP][Flat_index]
    Extents ext;                        // Extents struct defined in strides.hpp

    AETHER_INLINE std::size_t idx(int i, int j, int k) const {return ext.index(i,j,k);} // Contains the boundary checking built into Extents 
    

};


}


