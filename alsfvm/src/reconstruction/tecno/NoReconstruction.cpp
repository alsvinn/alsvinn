#include "alsfvm/reconstruction/tecno/NoReconstruction.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
namespace alsfvm {
namespace reconstruction {
namespace tecno {
void NoReconstruction::performReconstruction(const volume::Volume& leftInput,
    const volume::Volume& rightInput,
    size_t,
    volume::Volume& leftOut,
    volume::Volume& rightOut) {


    for (size_t var = 0; var < leftInput.getNumberOfVariables(); var++) {
        auto pointerLeftIn = leftInput.getScalarMemoryArea(var)->getPointer();
        auto pointerRightIn = rightInput.getScalarMemoryArea(var)->getPointer();
        auto pointerLeft = leftOut.getScalarMemoryArea(var)->getPointer();
        auto pointerRight = rightOut.getScalarMemoryArea(var)->getPointer();

        volume::for_each_cell_index(leftInput,
        [&](size_t index) {

            pointerLeft[index] = pointerLeftIn[index];
            pointerRight[index] = pointerRightIn[index];
        });
    }



}
}
}
}
