#pragma once
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
namespace alsfvm {
namespace reconstruction {
namespace tecno {

//! Does no reconstruction, just copies the variables to the new struct.
class NoReconstructionCUDA : public TecnoReconstruction {
    public:

        //! Copies the variables to the new arrays.
        //!
        //! @param[in] leftInput the left values to use for reconstruction
        //! @param[in] rightInput the right values to use for reconstruction
        //! @param[in] direction the direction (0=x, 1=y, 2=y)
        //! @param[out] leftOutput at the end, should contain reconstructed values
        //! @param[out] rightOutput at the end, should contain the reconstructed values
        virtual void performReconstruction(const volume::Volume& leftInput,
            const volume::Volume& rightInput,
            size_t direction,
            volume::Volume& leftOutput,
            volume::Volume& rightOutput);


        virtual size_t getNumberOfGhostCells() const {
            return 1;
        }
};
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
