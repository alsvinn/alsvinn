#include "alsfvm/diffusion/NoDiffusion.hpp"

namespace alsfvm {
namespace diffusion {
void NoDiffusion::applyDiffusion(volume::Volume& outputVolume,
    const volume::Volume& conservedVolume) {
    // Empty, doesn't do anything
}


/// Gets the total number of ghost cells this diffusion needs,
/// this is typically governed by reconstruction algorithm.
size_t NoDiffusion::getNumberOfGhostCells() const {
    return 0;
}
}
}
