#pragma once
#include "alsfvm/cuda/CudaBuffer.hpp"
namespace alsfvm {
namespace cuda {

///
/// 1D cuda buffer
///
template<typename T>
class CudaBuffer1D : public CudaBuffer < T > {
    public:
        ///
        /// \param nx the number of Ts in x direction
        ///
        CudaBuffer1D(size_t nx);
        virtual ~CudaBuffer1D();


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
