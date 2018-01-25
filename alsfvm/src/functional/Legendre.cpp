#include "alsfvm/functional/Legendre.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <boost/math/special_functions/legendre.hpp>


namespace alsfvm { namespace functional {

Legendre::Legendre(const Functional::Parameters &parameters)
    : minValue(parameters.getDouble("minValue")), maxValue(parameters.getDouble("maxValue")),
      degree(parameters.getInteger("degree"))
{

}

void Legendre::operator()(volume::Volume &conservedVolumeOut,
                          volume::Volume &extraVolumeOut,
                          const volume::Volume &conservedVolumeIn,
                          const volume::Volume &extraVolumeIn,
                          const real weight,
                          const grid::Grid &grid)
{
    const auto lengths = grid.getCellLengths();
    const real dxdydz = lengths.x*lengths.y*lengths.z;

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();
    for(size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
        real integral = 0.0;

        volume::for_each_cell_index(conservedVolumeIn, [&](size_t i) {
            const real value = (conservedVolumeIn.getScalarMemoryArea(var)->getPointer()[i]-minValue)/(maxValue-minValue);
            integral += boost::math::legendre_p(degree, value) * dxdydz;
        }, ghostCells, ghostCells);

        conservedVolumeOut.getScalarMemoryArea(var)->getPointer()[0] += weight*integral;
    }


    for(size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
        real integral = 0.0;

        volume::for_each_cell_index(conservedVolumeIn,  [&](size_t i) {
            const real value = (extraVolumeIn.getScalarMemoryArea(var)->getPointer()[i]-minValue)/(maxValue-minValue);
            integral += boost::math::legendre_p(degree, value) * dxdydz;
        }, ghostCells, ghostCells);

        extraVolumeOut.getScalarMemoryArea(var)->getPointer()[0] += weight*integral;
    }
}

ivec3 Legendre::getFunctionalSize() const
{
    return ivec3{1,1,1};
}

REGISTER_FUNCTIONAL(cpu, legendre, Legendre)
}
}
