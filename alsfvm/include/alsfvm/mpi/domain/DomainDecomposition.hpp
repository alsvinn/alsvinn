#pragma once
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/mpi/domain/DomainInformation.hpp"
#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {
//! Abstract base class to do domain decomposition
class DomainDecomposition {
public:
    virtual ~DomainDecomposition() {}


    //! Decomposes the grid. The returned object is the local information
    //! for this node.
    //!
    //! @param configuration the configuration
    //! @param grid the grid
    virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
        const grid::Grid& grid) = 0;
};
} // namespace domain
} // namespace mpi
} // namespace alsfvm
