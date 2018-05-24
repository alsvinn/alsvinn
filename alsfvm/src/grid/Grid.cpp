/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/grid/Grid.hpp"
#include <algorithm>
#include <iostream>
namespace alsfvm {
namespace grid {

namespace {
rvec3 computeCellLengths(const rvec3& origin, const rvec3& top,
    const ivec3 dimensions) {
    int dimensionX = dimensions[0];
    int dimensionY = dimensions[1];
    int dimensionZ = dimensions[2];

    auto lengths = (top - origin) / rvec3(dimensionX, dimensionY, dimensionZ);

    if (dimensionY == 1) {
        lengths[1] = 1;
    }

    if (dimensionZ == 1) {
        lengths[2] = 1;
    }

    return lengths;
}
}

Grid::Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
    const std::array<boundary::Type, 6>& boundaryConditions
)
    : Grid(origin, top, dimensions, boundaryConditions, {
    0, 0, 0
}, dimensions) {

}

///
/// Constructs the Grid
/// \param origin the origin point of the grid (the smallest point in lexicographical order)
/// \param top the top right corner of the grid (maximum point in lexicographical order)
/// \param dimensions the dimensions of the grid (in number of cells in each direction)
///
Grid::Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
    const std::array<boundary::Type, 6>& boundaryConditions,
    const ivec3& globalPosition,
    const ivec3& globalSize)

    : Grid(origin, top, dimensions, boundaryConditions, globalPosition, globalSize,
          computeCellLengths(origin, top, dimensions))

{

}

Grid::Grid(rvec3 origin,
    rvec3 top,
    ivec3 dimensions,
    const std::array<boundary::Type, 6>& boundaryConditions,
    const ivec3& globalPosition,
    const ivec3& globalSize,
    const rvec3& cellLengths)
    : origin(origin), top(top), dimensions(dimensions),
      cellLengths(cellLengths),
      boundaryConditions(boundaryConditions),
      globalPosition(globalPosition),
      globalSize(globalSize)

{
    // Create the cell midpoints
    cellMidpoints.resize(dimensions.x * dimensions.y * dimensions.z);

    for (int z = 0; z < dimensions.z; z++) {
        for (int y = 0; y < dimensions.y; y++) {
            for (int x = 0; x < dimensions.x; x++) {
                rvec3 position = origin
                    + rvec3(cellLengths.x * x, cellLengths.y * y, cellLengths.z * z)
                    + cellLengths / real(2.0);

                cellMidpoints[z * dimensions.x * dimensions.y + y * dimensions.x + x] =
                    position;
            }
        }
    }
}

Grid::Grid(rvec3 origin,
    rvec3 top,
    ivec3 dimensions,
    const std::array<boundary::Type, 6>& boundaryConditions,
    const ivec3& globalPosition,
    const ivec3& globalSize,
    const rvec3& cellLengths,
    const std::vector<rvec3>& cellMidpointsGlobal)
    : origin(origin), top(top), dimensions(dimensions),
      cellLengths(cellLengths),
      boundaryConditions(boundaryConditions),
      globalPosition(globalPosition),
      globalSize(globalSize)

{
    // Create the cell midpoints
    cellMidpoints.resize(dimensions.x * dimensions.y * dimensions.z);

    for (int z = 0; z < dimensions.z; z++) {
        for (int y = 0; y < dimensions.y; y++) {
            for (int x = 0; x < dimensions.x; x++) {
                const int indexGlobal = (z + globalPosition.z) * globalSize.x * globalSize.y
                    + (y + globalPosition.y) * globalSize.x
                    + (x + globalPosition.x);
                rvec3 position = cellMidpointsGlobal[indexGlobal];

                cellMidpoints[z * dimensions.x * dimensions.y + y * dimensions.x + x] =
                    position;
            }
        }
    }

}

///
/// Gets the origin point
/// \returns the origin point
///
rvec3 Grid::getOrigin() const {
    return origin;
}

///
/// Gets the top point
/// \returns the top point
///
rvec3 Grid::getTop() const {
    return top;
}

///
/// Gets the dimensions
/// \returns the dimensions (number of cells in each direction)
///
ivec3 Grid::getDimensions() const {
    return dimensions;
}

size_t Grid::getActiveDimension() const {
    if (dimensions.z > 1) {
        return 3;
    } else if (dimensions.y > 1) {
        return 2;
    } else {
        return 1;
    }
}

///
/// Gets the cell lengths in each direction
///
rvec3 Grid::getCellLengths() const {
    return cellLengths;
}

const std::vector<rvec3>& Grid::getCellMidpoints() const {
    return cellMidpoints;
}

boundary::Type Grid::getBoundaryCondition(int side) const {
    return boundaryConditions[side];
}

std::array<boundary::Type, 6> Grid::getBoundaryConditions() const {
    return boundaryConditions;
}

ivec3 Grid::getGlobalPosition() const {
    return globalPosition;
}

ivec3 Grid::getGlobalSize() const {
    return globalSize;
}
}
}
