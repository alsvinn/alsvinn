#include "alsfvm/functional/Legendre.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <boost/math/special_functions/legendre.hpp>


namespace alsfvm { namespace functional {

Legendre::Legendre(const Functional::Parameters &parameters)
    : minValue(parameters.getDouble("minValue")), maxValue(parameters.getDouble("maxValue")),
      degree(parameters.getInteger("degree"))
{
    if (parameters.contains("variables")) {
        for (auto variable : parameters.getStringVectorFromString("variables")) {
            variables.push_back(variable);
        }
    }
}

void Legendre::operator()(volume::Volume &conservedVolumeOut,
                          volume::Volume &extraVolumeOut,
                          const volume::Volume &conservedVolumeIn,
                          const volume::Volume &extraVolumeIn,
                          const real weight,
                          const grid::Grid &grid)
{
    if (variables.size() == 0) {
        for(size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(conservedVolumeIn.getName(var));
        }

        for(size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(extraVolumeIn.getName(var));
        }

    }
    const auto lengths = grid.getCellLengths();
    const real dxdydz = lengths.x*lengths.y*lengths.z;

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();

    for (const std::string& variableName : variables) {
        if (conservedVolumeIn.hasVariable(variableName)) {
            real integral = 0.0;

            volume::for_each_cell_index(conservedVolumeIn, [&](size_t i) {
                const real value = (conservedVolumeIn.getScalarMemoryArea(variableName)->getPointer()[i]-minValue)/(maxValue-minValue);
                integral += boost::math::legendre_p(degree, value) * dxdydz;
            }, ghostCells, ghostCells);

            conservedVolumeOut.getScalarMemoryArea(variableName)->getPointer()[0] += weight*integral;
        }
        else if (extraVolumeIn.hasVariable(variableName)) {
            for(size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
                real integral = 0.0;

                volume::for_each_cell_index(conservedVolumeIn,  [&](size_t i) {
                    const real value = (extraVolumeIn.getScalarMemoryArea(var)->getPointer()[i]-minValue)/(maxValue-minValue);
                    integral += boost::math::legendre_p(degree, value) * dxdydz;
                }, ghostCells, ghostCells);

                extraVolumeOut.getScalarMemoryArea(var)->getPointer()[0] += weight*integral;
            }
        } else {
            THROW("Unknown variable name given to Legendre functional: " << variableName);
        }
    }

}

ivec3 Legendre::getFunctionalSize(const grid::Grid& grid) const
{
    return {1,1,1};
}

REGISTER_FUNCTIONAL(cpu, legendre, Legendre)
}
}
