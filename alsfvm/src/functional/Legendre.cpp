#include "alsfvm/functional/Legendre.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
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
    real lengths = grid.getCellLengths();
    real dxdydz = lengths.x*lengths.y*lengths.z;


    for(int var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
        real integral = 0.0;

        volume::for_each_internal_volume_index(conservedVolumeIn, 0, [&](size_t, size_t i, size_t) {
            const real value = (conservedVolumeIn.getScalarMemoryArea(var)->getPointer()[i]-minValue)/(maxValue-minValue);
            integral += boost::math::legendre_p(degree, value) * dxdydz;
        });

        conservedVolumeOut.getScalarMemoryArea(var)->getPointer()[0] += weight*integral;
    }


    for(int var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
        real integral = 0.0;

        volume::for_each_internal_volume_index(extraVolumeIn, 0, [&](size_t, size_t i, size_t) {
            const real value = (extraVolumeIn.getScalarMemoryArea(var)->getPointer()[i]-minValue)/(maxValue-minValue);
            integral += boost::math::legendre_p(degree, value) * dxdydz;
        });

        extraVolumeOut.getScalarMemoryArea(var)->getPointer()[0] += weight*integral;
    }
}

ivec3 Legendre::getFunctionalSize()
{
    return ivec3{1,1,1}
}

}
}
