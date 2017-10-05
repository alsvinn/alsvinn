#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/mpi/CellExchanger.hpp"

namespace alsfvm { namespace mpi { namespace domain { 

    //! Contains information about the domain this processor is assigned, as well
    //! as the given neighbours
    class DomainInformation {
    public:
        DomainInformation(alsfvm::shared_ptr<grid::Grid> grid,
                          CellExchangerPtr cellExchanger);

        alsfvm::shared_ptr<grid::Grid> getGrid();

        CellExchangerPtr getCellExchanger();
    private:
        alsfvm::shared_ptr<grid::Grid> grid;
        CellExchangerPtr cellExchanger;

    };

    typedef alsfvm::shared_ptr<DomainInformation> DomainInformationPtr;
} // namespace domain
} // namespace mpi
} // namespace alsfvm
