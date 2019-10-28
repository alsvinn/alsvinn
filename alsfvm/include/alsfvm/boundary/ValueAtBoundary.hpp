#pragma once
#include "alsfvm/boundary/Type.hpp"
#include "alsfvm/memory/View.hpp"
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace boundary {

//! Simple templated class to get the value at the boundary
template<Type BoundaryType>
class ValueAtBoundary {
public:

};

template<>
class ValueAtBoundary<PERIODIC> {
public:
    template<class T>
    __device__ __host__ static T getValueAtBoundary(
        const memory::View<T>& view,
        ivec3 discretePosition,
        ivec3 numberOfCellsWithoutGhostCells,
        ivec3 numberGhostCells) {

        const auto positivePosition = makePositive(discretePosition,
                numberOfCellsWithoutGhostCells);

        const auto moduloPosition = ivec3{positivePosition.x % numberOfCellsWithoutGhostCells.x,
                       positivePosition.y % numberOfCellsWithoutGhostCells.y,
                       positivePosition.z % numberOfCellsWithoutGhostCells.z};

        const auto positionPeriodic = moduloPosition + numberGhostCells;

        return view.at(view.index(positionPeriodic.x, positionPeriodic.y,
                    positionPeriodic.z));
    }

private:
    static __device__ __host__ int makePositive(int position, int N) {
        while (position < 0) {
            position += N;
        }

        return position;

    }
    static __device__ __host__ ivec3 makePositive(ivec3 position,
        ivec3 numberOfCells) {
        return ivec3{makePositive(position.x, numberOfCells.x),
                makePositive(position.y, numberOfCells.y),
                makePositive(position.z, numberOfCells.z)};

    }
};



template<>
class ValueAtBoundary<NEUMANN> {
public:
    template<class T>
    __device__ __host__ static T getValueAtBoundary(
        const memory::View<T>& view,
        ivec3 discretePosition,
        ivec3 numberOfCellsWithoutGhostCells,
        ivec3 numberGhostCells) {
        // fixes for cuda.
        using namespace std;

        const auto innerPosition = ivec3{
            max(0, min(discretePosition.x, numberOfCellsWithoutGhostCells.x)),
            max(0, min(discretePosition.y, numberOfCellsWithoutGhostCells.y)),
            max(0, min(discretePosition.z, numberOfCellsWithoutGhostCells.z))
        };

        const auto positionWithGhostCells = innerPosition + numberGhostCells;
        return view.at(view.index(positionWithGhostCells.x, positionWithGhostCells.y,
                    positionWithGhostCells.z));
    }

};
} // namespace boundary
} // namespace alsfvm
