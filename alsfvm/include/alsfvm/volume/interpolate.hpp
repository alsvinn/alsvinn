#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsutils/error/Exception.hpp"


namespace alsfvm {
namespace volume {

template<size_t dimension>
void interpolate(memory::View<real>& out, memory::View<const real>& in,
    size_t x, size_t y, size_t z);

template<>
void interpolate<1>(memory::View<real>& out, memory::View<const real>& in,
    size_t x, size_t y, size_t z) {
    out.at(x, y, z) = (in.at(2 * x, y, z) + in.at(2 * x + 1, y, z)) / 2.0;
}

template<>
void interpolate<2>(memory::View<real>& out, memory::View<const real>& in,
    size_t x, size_t y, size_t z) {
    out.at(x, y, z) = (in.at(2 * x, 2 * y, z) + in.at(2 * x + 1, 2 * y, z)
            + in.at(2 * x, 2 * y + 1, z) + in.at(2 * x + 1, 2 * y + 1, z)) / 4.0;
}

template<>
void interpolate<3>(memory::View<real>& out, memory::View<const real>& in,
    size_t x, size_t y, size_t z) {
    out.at(x, y, z) = 0;

    // Do this with for-loop because there are just too many combinations
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                out.at(x, y, z) += in.at(2 * x + k, 2 * y + j, 2 * z + i);
            }
        }
    }

    out.at(x, y, z) /= 16;
}

template<size_t dimension>
inline void interpolate(Volume& out, const Volume& in) {
    if (out.getNumberOfXCells() != in.getNumberOfXCells() / 2) {
        THROW("Currently we only support doing interpolation with the ration 2 to 1.");
    }

    size_t nx = out.getTotalNumberOfXCells();
    size_t ny = out.getTotalNumberOfYCells();
    size_t nz = out.getTotalNumberOfZCells();

    size_t ng = out.getNumberOfXGhostCells();
    bool hasZ = dimension > 2;
    bool hasY = dimension > 1;

    for (size_t var = 0; var < out.getNumberOfVariables(); ++var) {
        auto viewOut = out.getScalarMemoryArea(var)->getView();
        auto viewIn = in.getScalarMemoryArea(var)->getView();

        for (size_t z = hasZ * ng; z < nz - hasZ * ng ; ++z) {
            for (size_t y = hasY * ng; y < ny - hasY * ng; ++y) {
                for (size_t x = ng; x < nx - ng; ++x) {
                    interpolate<dimension>(viewOut, viewIn, x, y, z);
                }
            }
        }
    }


}
}
}
