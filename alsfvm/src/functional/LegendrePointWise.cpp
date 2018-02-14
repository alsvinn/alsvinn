#include "alsfvm/functional/LegendrePointWise.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <boost/math/special_functions/legendre.hpp>


namespace alsfvm {
namespace functional {

LegendrePointWise::LegendrePointWise(const Functional::Parameters& parameters)
    : minValue(parameters.getDouble("minValue")),
      maxValue(parameters.getDouble("maxValue")),
      degree(parameters.getInteger("degree")) {
    if (parameters.contains("variables")) {
        for (auto variable : parameters.getStringVectorFromString("variables")) {
            variables.push_back(variable);
        }
    }
}

void LegendrePointWise::operator()(volume::Volume& conservedVolumeOut,
    volume::Volume& extraVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const volume::Volume& extraVolumeIn,
    const real weight,
    const grid::Grid& grid) {


    if (variables.size() == 0) {
        for (size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(conservedVolumeIn.getName(var));
        }

        for (size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(extraVolumeIn.getName(var));
        }

    }

    const auto lengths = grid.getCellLengths();

    if (lengths.z > 1 || lengths.y == 1) {
        THROW("For now, Legendre polynomials only support 2d, givne dimensions " <<
            lengths);
    }

    const real dxdydz = lengths.x * lengths.y * lengths.z;

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();
    const auto origin = grid.getOrigin();
    const auto top = grid.getTop();
    const auto sides = top - origin;

    const auto innerSize = conservedVolumeIn.getInnerSize();

    for (const std::string& variableName : variables) {
        if (conservedVolumeIn.hasVariable(variableName)) {


            auto viewIn = conservedVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = conservedVolumeOut.getScalarMemoryArea(variableName)->getView();

            for (int k = 0; k < innerSize.z; ++k) {
                for (int j = 0; j < innerSize.y; ++j) {
                    for (int i = 0; i < innerSize.x; ++i) {
                        const real value = (viewIn.at(i + ghostCells.x, j + ghostCells.y,
                                    k + ghostCells.z) - minValue) / (maxValue - minValue);
                        viewOut.at(i, j, k) += boost::math::legendre_p(degree, value) * weight;
                    }
                }


            }


        } else if (extraVolumeIn.hasVariable(variableName)) {

            auto viewIn = extraVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = extraVolumeOut.getScalarMemoryArea(variableName)->getView();

            for (int k = 0; k < innerSize.z; ++k) {
                for (int j = 0; j < innerSize.y; ++j) {
                    for (int i = 0; i < innerSize.x; ++i) {
                        const real value = (viewIn.at(i + ghostCells.x, j + ghostCells.y,
                                    k + ghostCells.z) - minValue) / (maxValue - minValue);
                        viewOut.at(i, j, k) += boost::math::legendre_p(degree, value) * weight;
                    }
                }
            }



        } else {
            THROW("Unknown variable name given to LegendrePointWise functional: " <<
                variableName);
        }
    }

}

ivec3 LegendrePointWise::getFunctionalSize(const grid::Grid& grid) const {
    return grid.getDimensions();
}

REGISTER_FUNCTIONAL(cpu, legendre_pointwise, LegendrePointWise)
}
}
