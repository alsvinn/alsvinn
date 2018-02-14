#include "alsfvm/mpi/CellExchanger.hpp"

namespace alsfvm {
namespace mpi {

real CellExchanger::adjustWaveSpeed(real waveSpeed) {
    return this->max(waveSpeed);
}


}
}
