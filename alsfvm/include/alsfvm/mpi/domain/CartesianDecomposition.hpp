#pragma once
#include "alsfvm/mpi/domain/DomainDecompositionParameters.hpp"
#include "alsfvm/mpi/domain/DomainDecomposition.hpp"
namespace alsfvm {
namespace mpi {
namespace domain {

//! Performs domain decomposition on a regular cartesian grid
class CartesianDecomposition : public DomainDecomposition {
public:

    //! Constructs a new decomposition with the parameters,
    //! uses the parameters for nx, ny, nz
    //!
    //! @param parameters used for nx, ny, nz
    CartesianDecomposition(const DomainDecompositionParameters& parameters);

    //! Constructs a new decomposition with the parameters,
    //! uses the parameters for nx, ny, nz
    //!
    //! @param nx number of cpus in x direction
    //! @param ny number of cpus in y direction
    //! @param nz number of cpus in z direction
    CartesianDecomposition(int nx, int ny, int nz);


    //! Decomposes the domain
    //!
    //! @param configuratino the given mpi configuration
    //! @param grid the whole grid to decompose
    //! @return the domain information, containing the cell exchanger and
    //!         the new grid.
    virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
        const grid::Grid& grid
    ) override;

private:
    const ivec3 numberOfProcessors;
};
} // namespace domain
} // namespace mpi
} // namespace alsfvm
