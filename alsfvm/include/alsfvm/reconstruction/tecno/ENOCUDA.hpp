#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
namespace alsfvm { namespace reconstruction { namespace tecno { 

    template<int order>
    class ENOCUDA : public TecnoReconstruction {
    public:

        ENOCUDA(alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                       size_t nx, size_t ny, size_t nz);

        //! Applies the reconstruction.
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


        ///
        /// \brief getNumberOfGhostCells returns the number of ghost cells we need
        ///        for this computation
        /// \return order.
        ///
        virtual size_t getNumberOfGhostCells() const;


    private:

        //! Creates the divided differences arrays
        //! @param nx number of x cells
        //! @param ny number of y cells
        //! @param nz number of z cells
        void makeDividedDifferenceArrays(size_t nx, size_t ny, size_t nz);

        //! Computes the divded differences. This is specially made for the
        //! tecno scheme. See the tecno paper for more details
        //!
        void computeDividedDifferences(const memory::Memory<real>& leftInput,
                                       const memory::Memory<real>& rightInput,
                                       const ivec3& direction,
                                       size_t level,
                                       memory::Memory<real>& output);


        alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
        // For each level l, this will contain the divided differences for that
        // level.
        std::array<alsfvm::shared_ptr<memory::Memory<real> >, order - 1> dividedDifferences;

    };
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
