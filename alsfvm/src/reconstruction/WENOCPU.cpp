#include "alsfvm/reconstruction/WENOCPU.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include <array>
#include <limits>
#include "alsfvm/error/Exception.hpp"
#include <cassert>
#include <type_traits>



namespace alsfvm {

namespace reconstruction {


namespace {


template<int i, int order>
typename std::enable_if<i == -1>::type computeAlpha(std::array<real, 2 * order - 1>& stencil,
                                     real& sumLeft, real& sumRight,
                                     std::array<real, order>& left, std::array<real, order>& right)
{
    // empty
}

    template<int i, int order>
    typename std::enable_if<i != -1>::type computeAlpha(
            std::array<real, 2 * order - 1>& stencil,
            real& sumLeft, real& sumRight,
            std::array<real, order>& left,
            std::array<real, order>& right)
    {
        const real epsilon = WENOCoefficients<order>::epsilon;
        const real beta = WENOCoefficients<order>::template computeBeta<i>(stencil);
        right[i] = WENOCoefficients<order>::coefficients[i] / std::pow(beta + epsilon, 2);
        sumRight += right[i];

        left[i] = WENOCoefficients<order>::coefficients[order - 1 - i] / std::pow(beta + epsilon, 2);
        sumLeft += left[i];

       computeAlpha<i - 1, order>(stencil, sumLeft, sumRight, left, right);
    }




}

template<int order>
WENOCPU<order>::WENOCPU() {
    //empty
}

template<int order>
void WENOCPU<order>::performReconstruction(const volume::Volume& inputVariables,
                                           size_t direction,
                                           size_t indicatorVariable,
                                           volume::Volume& leftOut,
                                           volume::Volume& rightOut)
{

    // We often do compute order-1.
    static_assert(order > 0, "Can not do WENO reconstruction of order 0.");

    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }
    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    const size_t numberOfVariables = inputVariables.getNumberOfVariables();

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.
    const size_t nx = inputVariables.getTotalNumberOfXCells();
    const size_t ny = inputVariables.getTotalNumberOfYCells();
    const size_t nz = inputVariables.getTotalNumberOfZCells();

    // Sanity check, we need at least ONE point in the interior.
    assert(int(nx) > 2 * directionVector.x * order);
    assert((directionVector.y == 0) || (int(ny) > 2 * directionVector.y * order));
    assert((directionVector.z == 0) || (int(nz) > 2 * directionVector.z * order));

    const size_t startX = directionVector.x * (order - 1);
    const size_t startY = directionVector.y * (order - 1);
    const size_t startZ = directionVector.z * (order - 1);

    const size_t endX = nx - directionVector.x * (order - 1);
    const size_t endY = ny - directionVector.y * (order - 1);
    const size_t endZ = nz - directionVector.z * (order - 1);



    std::vector<const real*> pointersIn(inputVariables.getNumberOfVariables());
    std::vector<real*> pointersOutLeft(inputVariables.getNumberOfVariables());
    std::vector<real*> pointersOutRight(inputVariables.getNumberOfVariables());

    for (size_t var = 0; var < inputVariables.getNumberOfVariables(); var++) {
        pointersIn[var] = inputVariables.getScalarMemoryArea(var)->getPointer();
        pointersOutLeft[var] = leftOut.getScalarMemoryArea(var)->getPointer();
        pointersOutRight[var] = rightOut.getScalarMemoryArea(var)->getPointer();
    }


    const real* pointerInWeight = inputVariables.getScalarMemoryArea(indicatorVariable)->getPointer();

    for (size_t z = startZ; z < endZ; z++) {
        for (size_t y = startY; y < endY; y++) {
            #pragma omp parallel for
            for (int x = startX; x < endX; x++) {
                const size_t outIndex = z*nx*ny + y * nx + x;




                // First we need to find alpha and beta.
                std::array<real, 2*order - 1> stencil;
                for (int i = -order + 1; i < order; i++) {
                    const size_t index = (z + i * directionVector.z) * nx * ny
                            + (y + i*directionVector.y) * nx
                            + (x + i*directionVector.x);

                    stencil[i + order - 1] = pointerInWeight[index];
                }

                std::array<real, order> alphaRight;
                std::array<real, order> alphaLeft;
                real alphaRightSum = 0.0;
                real alphaLeftSum = 0.0;

                static_assert(order < 4 && order > 1, "So far, we only support order = 2 or order = 3");

                computeAlpha<int(order) - 1, int(order)>(stencil,
                                                         alphaLeftSum,
                                                         alphaRightSum,
                                                         alphaLeft,
                                                         alphaRight);

                for(size_t var = 0; var < numberOfVariables; var++) {
                    real leftWenoValue = 0.0;
                    real rightWenoValue = 0.0;
                    // Loop through all stencils (shift = r)
                    for (int shift = 0; shift < order; shift++) {

                        auto coefficientsRight = ENOCoeffiecients<order>::coefficients[shift + 1];
                        auto coefficientsLeft = ENOCoeffiecients<order>::coefficients[shift];
                        real leftValue = 0.0;
                        real rightValue = 0.0;
                        for (int j = 0; j < order; j++) {

                            const size_t index = (z - (shift - j)*directionVector.z) * nx * ny
                                    + (y - (shift - j)*directionVector.y) * nx
                                    + (x - (shift - j)*directionVector.x);


                            const real value = pointersIn[var][index];
                            leftValue += coefficientsLeft[j] * value;
                            rightValue += coefficientsRight[j] * value;
                        }
                        leftWenoValue += leftValue * (alphaLeft[shift] / alphaLeftSum);
                        rightWenoValue += rightValue * (alphaRight[shift] / alphaRightSum);
                    }

                    pointersOutLeft[var][outIndex] = leftWenoValue;
                    pointersOutRight[var][outIndex] = rightWenoValue;

                }
            }
        }
    }

}

template class WENOCPU<2>;
template class WENOCPU<3>;
}
}
