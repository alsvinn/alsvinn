#include "alsfvm/functional/BoundedVariation.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm {
namespace functional {

BoundedVariation::BoundedVariation(const Functional::Parameters& parameters)
    :      degree(parameters.getInteger("degree")) {

}

void BoundedVariation::operator()(volume::Volume& conservedVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const real weight,
    const grid::Grid& grid) {

    const auto lengths = grid.getCellLengths();

    if (lengths.z > 1) {
        THROW("For now, BoundedVariation  only support 2d and 1d, given dimensions "
            <<
            lengths);
    }

    const ivec3 start = conservedVolumeIn.getNumberOfGhostCells();
    const ivec3 end = conservedVolumeIn.getSize() -
        conservedVolumeIn.getNumberOfGhostCells();

    for (size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
        conservedVolumeOut.getScalarMemoryArea(
            var)->getPointer()[0] += weight * conservedVolumeIn.getScalarMemoryArea(
                    var)->getTotalVariation(degree,
                    start,
                    end);
    }
}

ivec3 BoundedVariation::getFunctionalSize(const grid::Grid& grid) const {
    return {1, 1, 1};
}

std::string BoundedVariation::getPlatformToAllocateOn(const std::string&
    platform)
const {
    return "cpu";
}

REGISTER_FUNCTIONAL(cpu, bv, BoundedVariation)
REGISTER_FUNCTIONAL(cuda, bv, BoundedVariation)
}
}
