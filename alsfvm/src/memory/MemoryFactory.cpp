#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#ifdef  ALSVINN_HAVE_CUDA
#include "alsfvm/cuda/CudaMemory.hpp"
#endif
namespace alsfvm {
	namespace memory {


		/// 
		/// \param memoryName the name of the memory implementation to use (eg. HostMemory, CudaMemory, ...)
		/// \param deviceConfiguration the deviceConfiguration to use (this is mostly only relevant for GPU, on CPU it can be empty)
		///
		MemoryFactory::MemoryFactory(const std::string& memoryName, std::shared_ptr<DeviceConfiguration>& deviceConfiguration)
			: memoryName(memoryName), deviceConfiguration(deviceConfiguration)
		{
		}

		///
		/// Creates scalar memory of the given size
        /// \param nx the number of real numbers to store in X direction
        /// \param ny the number of real numbers to store in Y direction
        /// \param nz the number of real numbers to store in Z direction
        /// \note nx, ny, nz is in number of reals
		/// \returns the pointer to the newly created memory area.
		///
        std::shared_ptr<Memory<real> >
            MemoryFactory::createScalarMemory(size_t nx, size_t ny, size_t nz) {
            if (memoryName == "HostMemory") {
                return std::shared_ptr<Memory<real> >(new HostMemory<real>(nx, ny, nz));
            }
            else if (memoryName == "CudaMemory") {
#ifdef ALSVINN_HAVE_CUDA
                return std::shared_ptr<Memory<real> >(new CudaMemory(nx,ny, nz));
#else
                THROW("CUDA is not enabled for this build");
#endif
            } else {
                THROW("Unknown memory type " << memoryName);
            }
		}
	}
}
