#pragma once
#include "alsfvm/cuda/CudaBuffer.hpp"
namespace alsfvm {
namespace cuda {

///
/// 1D cuda buffer
///
template<class T>
class CudaBuffer3D : public CudaBuffer < T > {
    public:
        ///
        /// \param nx the number of Ts in x direction
        /// \param ny the number of Ts in y direction
        /// \param nz the number of Ts in z direction
        ///
        CudaBuffer3D(size_t nx, size_t ny, size_t nz);
        virtual ~CudaBuffer3D();


        ///
        /// \returns the pointer to the data
        ///
        virtual T* getPointer();

        ///
        /// \returns the pointer to the data
        ///
        virtual const T* getPointer() const;


    private:
        T* memoryPointer;
};
}
}