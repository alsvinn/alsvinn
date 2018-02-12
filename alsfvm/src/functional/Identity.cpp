#include "alsfvm/functional/Identity.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm { namespace functional {

Identity::Identity(const Functional::Parameters &parameters)
{

}

void Identity::operator()(volume::Volume &conservedVolumeOut,
                          volume::Volume &extraVolumeOut,
                          const volume::Volume &conservedVolumeIn,
                          const volume::Volume &extraVolumeIn,
                          const real weight,
                          const grid::Grid &grid)
{

    const auto lengths = grid.getCellLengths();

    if (lengths.z > 1 || lengths.y == 1) {
        THROW("For now, Legendre polynomials only support 2d, givne dimensions " << lengths);
    }


    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();


    const auto innerSize = conservedVolumeIn.getInnerSize();


    for(size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {


        auto viewIn = conservedVolumeIn.getScalarMemoryArea(var)->getView();
        auto viewOut = conservedVolumeOut.getScalarMemoryArea(var)->getView();
        for (int k = 0; k < innerSize.z; ++k) {
            for (int j = 0; j < innerSize.y; ++j) {
                for (int i = 0; i < innerSize.x; ++i) {
                    const real value = viewIn.at(i+ghostCells.x, j+ghostCells.y, k + ghostCells.z);
                    viewOut.at(i,j,k) += weight*value;
                }
            }


        }


    }
    for(size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {



        auto viewIn = extraVolumeIn.getScalarMemoryArea(var)->getView();
        auto viewOut = extraVolumeOut.getScalarMemoryArea(var)->getView();
        for (int k = 0; k < innerSize.z; ++k) {
            for (int j = 0; j < innerSize.y; ++j) {
                for (int i = 0; i < innerSize.x; ++i) {
                    const real value = viewIn.at(i+ghostCells.x, j+ghostCells.y, k + ghostCells.z);
                    viewOut.at(i,j,k) += weight * value;
                }
            }
        }



    }
}

ivec3 Identity::getFunctionalSize(const grid::Grid &grid) const
{

}
REGISTER_FUNCTIONAL(cpu, identiy, Identity)
}
}
