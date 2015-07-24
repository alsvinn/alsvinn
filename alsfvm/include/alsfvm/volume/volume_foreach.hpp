#pragma once
#include "alsfvm/volume/Volume.hpp"
#include <functional>
#include <array>
///
/// This file contains for_each functions for volumes
///

namespace alsfvm {
namespace volume {

///
/// Loops through all possible cell indexes in a cache optimal manner.
/// Example:
/// \code{.cpp}
/// for_each_cell_index(someVolume, [](size_t index) {
///     std::cout << "index = " << index;
/// }):
/// \endcode
///
    template<class Function>
    inline void for_each_cell_index(const Volume& in, const Function& function) {
        const size_t nx = in.getNumberOfXCells();
        const size_t ny = in.getNumberOfYCells();
        const size_t nz = in.getNumberOfZCells();
        for(size_t k = 0; k < nz; k++) {
            for(size_t j = 0; j < ny; j++) {
                for(size_t i = 0; i < nz; i++) {
                    size_t index = k*nx*ny + j*ny + i;
                    function(index);
                }
            }
        }
    }


    template<class VariableStruct>
    inline VariableStruct expandVariableStruct(const std::array<const real*, 5>& in, size_t index) {
        return VariableStruct(in[0][index], in[1][index], in[2][index], in[3][index], in[4][index]);
    }

    template<class VariableStruct>
    inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 4>& out) {
        real* inAsRealPointer = (real*)&in;

        out[0][index] = inAsRealPointer[0];
        out[1][index] = inAsRealPointer[1];
        out[2][index] = inAsRealPointer[2];
        out[3][index] = inAsRealPointer[3];
    }


    template<class VariableStructIn, class VariableStructOut>
    inline void transform_volume(const Volume& in, Volume& out,
                                 const std::function<VariableStructOut(const VariableStructIn&)>& function) {

        std::array<const real*, sizeof(VariableStructIn)/sizeof(real)> pointersIn;
        for(size_t i = 0; i < in.getNumberOfVariables(); i++) {
            pointersIn[i] = in.getScalarMemoryArea(i)->getPointer();
        }

        std::array<real*, sizeof(VariableStructOut)/sizeof(real)> pointersOut;
        for(size_t i = 0; i < out.getNumberOfVariables(); i++) {
            pointersOut[i] = out.getScalarMemoryArea(i)->getPointer();
        }

        for_each_cell_index(in, [&](size_t index) {
            auto out = function(expandVariableStruct<VariableStructIn>(pointersIn, index));
            saveVariableStruct(out, index, pointersOut);

        });

    }
}
}
