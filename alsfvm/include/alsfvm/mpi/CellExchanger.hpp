#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/mpi/RequestContainer.hpp"

namespace alsfvm { namespace mpi { 

    //! Abstract base class for exchanging cells
    class CellExchanger {
    public:

        //! Does the exchange of cells between the processors.
        virtual RequestContainer exchangeCells(alsfvm::volume::Volume& outputVolume,
                                   const alsfvm::volume::Volume& inputVolume) = 0;
    };

    typedef alsfvm::shared_ptr<CellExchanger> CellExchangerPtr;
} // namespace mpi
} // namespace alsfvm
