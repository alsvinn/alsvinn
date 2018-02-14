#include "alsfvm/mpi/domain/DomainInformation.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {

DomainInformation::DomainInformation(alsfvm::shared_ptr<grid::Grid> grid,
    CellExchangerPtr cellExchanger)
    : grid(grid), cellExchanger(cellExchanger) {

}

alsfvm::shared_ptr<grid::Grid> DomainInformation::getGrid() {
    return grid;
}

CellExchangerPtr DomainInformation::getCellExchanger() {
    return cellExchanger;
}

}
}
}
