#pragma once
#include "alsfvm/mpi/CellExchanger.hpp"
#include "alsfvm/types.hpp"
#include "alsfvm/mpi/Configuration.hpp"
#include "alsfvm/mpi/MpiIndexType.hpp"
#include "alsfvm/mpi/Request.hpp"
#include "alsfvm/mpi/RequestContainer.hpp"

namespace alsfvm { namespace mpi { 

    //! Does the cell exchange for a cartesian grid.
    class CartesianCellExchanger : public CellExchanger {
    public:

        //! Constructs a new instance
        //!
        //! @param configuration a pointer to the current MPI configuration
        //! @param neighbours the list of processor neighbours for each side. Has
        //!                   the following format
        //!
        //! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
        //! -------|------------------|-----------------|-----------------
        //!    0   |       left       |     left        |    left
        //!    1   |       right      |     right       |    right
        //!    2   |     < not used > |     bottom      |    bottom
        //!    3   |     < not used > |     top         |    top
        //!    4   |     < not used > |   < not used >  |    front
        //!    5   |     < not used > |   < not used >  |    back
        CartesianCellExchanger(ConfigurationPtr& configuration,
                               const ivec6& neighbours);

        //! Does the exchange of data
        virtual RequestContainer exchangeCells(alsfvm::volume::Volume& outputVolume,
                           const alsfvm::volume::Volume& inputVolume) override;

        bool hasSide(int side) const;

        real max(real value) override;

    private:
        ConfigurationPtr configuration;

        const ivec6 neighbours;

        std::vector<MpiIndexTypePtr> datatypesReceive;
        std::vector<MpiIndexTypePtr> datatypesSend;

        void createDataTypes(const volume::Volume& volume);
        void createDataTypeSend(int side, const volume::Volume &volume);
        void createDataTypeReceive(int side, const volume::Volume &volume);
    };
} // namespace mpi
} // namespace alsfvm
