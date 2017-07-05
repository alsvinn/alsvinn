#pragma once
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
namespace alsfvm { namespace reconstruction { namespace tecno { 

//! Applies ENO reconstruction of order "order" (template argument),
//! according to the Tecno paper
//!
//! The reason we need a different class than for normal reconstruction
//! is that the input left and right values are a priori different.
//!
//! In other words, for tecno we reconstruct with
//!
//!    \f[u^l_i = R_{i+1/2}u_{i}\f]
//!    \f[u^r_i = R_{i-1/2}u_{i}\f]
//!
//! The reconstructions should be compatible with the Tecno paper
//!
//! Fjordholm, U. S., Mishra, S., & Tadmor, E. (2012). Arbitrarily high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws, 50(2), 544–573.
//!
//! See http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
    template<int order>
    class ENOCPU : public TecnoReconstruction {
    public:
        ENOCPU(alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
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

        void computeDividedDifferences(const memory::Memory<real>& leftInput,
                                       const memory::Memory<real>& rightInput,
                                       const ivec3& direction,
                                       size_t level,
                                       memory::Memory<real>& output);

        alsfvm::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactory;
        // For each level l, this will contain the divided differences for that
        // level.
        std::array<alsfvm::shared_ptr<memory::Memory<real> >, order - 1> dividedDifferences;
    };
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
