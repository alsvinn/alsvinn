#pragma once
#include "alsfvm/mpi/domain/DomainDecompositionParameters.hpp"
#include "alsfvm/mpi/domain/DomainDecomposition.hpp"
namespace alsfvm { namespace mpi { namespace domain { 

    //! Performs domain decomposition on a regular cartesian grid
    class CartesianDecomposition : public DomainDecomposition {
    public:
        CartesianDecomposition(const DomainDecompositionParameters& parameters);

        virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
                                            const grid::Grid &grid
                                            ) override;

    private:
        const ivec3 numberOfProcessors;
    };
} // namespace domain
} // namespace mpi
} // namespace alsfvm
