#pragma once
#include "alsfvm/memory/MemoryBase.hpp"
#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
#include "alsfvm/types.hpp"
#include <functional>
#include <map>

namespace alsfvm {
namespace memory {
    class MemoryFactory {
	public:
		///
		/// The base memory pointer
		///
		typedef alsfvm::shared_ptr < MemoryBase > MemoryPtr;
	

		/// 
		/// \param deviceConfiguration the deviceConfiguration to use (this is mostly only relevant for GPU, on CPU it can be empty)
		///
        MemoryFactory(alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

		///
		/// Creates scalar memory of the given size
        /// \param nx the number of real numbers to store in X direction
        /// \param ny the number of real numbers to store in Y direction
        /// \param nz the number of real numbers to store in Z direction
        /// \note nx, ny, nz is in number of reals
		/// \returns the pointer to the newly created memory area.
		///
        alsfvm::shared_ptr<Memory<real> > createScalarMemory(size_t nx, size_t ny, size_t nz);
	private:

		alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    };
}
}
