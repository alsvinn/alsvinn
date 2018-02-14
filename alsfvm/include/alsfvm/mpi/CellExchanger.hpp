#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/mpi/RequestContainer.hpp"
#include "alsfvm/integrator/WaveSpeedAdjuster.hpp"


namespace alsfvm {
namespace mpi {

//! Abstract base class for exchanging cells
class CellExchanger : public integrator::WaveSpeedAdjuster {
    public:

        virtual ~CellExchanger() {}
        //! Does the exchange of cells between the processors.
        virtual RequestContainer exchangeCells(alsfvm::volume::Volume& outputVolume,
            const alsfvm::volume::Volume& inputVolume) = 0;

        //! Does the maximum over all processors
        virtual real max(real number) = 0;

        //! Does the maximum over all wave speeds across processors
        real adjustWaveSpeed(real waveSpeed);

        virtual ivec6 getNeighbours() const = 0;
};

typedef alsfvm::shared_ptr<CellExchanger> CellExchangerPtr;
} // namespace mpi
} // namespace alsfvm
