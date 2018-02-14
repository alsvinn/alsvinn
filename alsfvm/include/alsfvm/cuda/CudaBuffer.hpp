#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace cuda {

///
/// A baseclass for holding CUDA memory
///
template<typename T>
class CudaBuffer {
    public:
        ///
        /// \param sizeInBytes the *total* size of the memory area (including pitch etc)
        /// \param nx number of cells in x direction (number of Ts)
        /// \param ny number of cells in y direction (number of Ts)
        /// \param nz number of cells in z direction(number of Ts)
        /// \param pitchX the pitch in X direction
        /// \param pitchY the pitch in Y direction
        ///
        /// \note see http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html
        ///       for how pitched memory works. This is relevant for 2D and 3D.
        ///
        CudaBuffer(size_t sizeInBytes,
            size_t nx,
            size_t ny,
            size_t nz,
            size_t pitchX,
            size_t pitchY);

        /// Virtual because we need to inherit from this
        virtual ~CudaBuffer();


        ///
        /// \returns the total size of the memory area in bytes
        ///
        virtual size_t getSizeInBytes() const;

        ///
        /// \returns the size (in number of T's)
        ///
        virtual size_t getSize() const;

        ///
        /// \note Do not use this for indexing!
        /// \returns the number of T's in X direction
        ///
        virtual size_t getXSize() const;


        ///
        /// \note Do not use this for indexing!
        /// \returns the number of T's in Y direction
        ///
        virtual size_t getYSize() const;


        ///
        /// \note Do not use this for indexing!
        /// \returns the number of T's in Z direction
        ///
        virtual size_t getZSize() const;

        ///
        /// \returns the pitch in Y direction  (in bytes!)
        /// \note Use for indexing by
        /// \code{.cpp}
        /// size_t index = i*getPitchY()*getPitchX() + j*getPitchX() + k;
        /// \endcode
        ///
        virtual size_t getPitchY() const;


        ///
        /// \returns the pitch in X direction (in bytes!)
        /// \note Use for indexing by
        /// \code{.cpp}
        /// size_t index = i*getPitchY()*getPitchX() + j*getPitchX() + k;
        /// \endcode
        ///
        virtual size_t getPitchX() const;


        ///
        /// \returns the pointer to the data
        ///
        virtual T* getPointer() = 0;

        ///
        /// \returns the pointer to the data
        ///
        virtual const T* getPointer() const = 0;


    private:
        size_t sizeInBytes;
        size_t nx;
        size_t ny;
        size_t nz;
        size_t pitchX;
        size_t pitchY;
};
}
}
