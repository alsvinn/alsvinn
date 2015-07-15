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
		typedef std::shared_ptr < MemoryBase > MemoryPtr;
	
		///
		/// The function definition for the memory constructors.
		///
		typedef std::function < MemoryPtr(size_t size, Types type,
			std::shared_ptr<DeviceConfiguration>& deviceConfiguration)> MemoryConstructor;

		///
		/// Add a constructor to the available constructors
		/// \note Use this function if you have written your own memory space that you would like
		/// to be made available.
		///
		static void addConstructor(const std::string& name, MemoryConstructor constructor);


		/// 
		/// \param memoryName the name of the memory implementation to use (eg. HostMemory, CudaMemory, ...)
		/// \param deviceConfiguration the deviceConfiguration to use (this is mostly only relevant for GPU, on CPU it can be empty)
		///
		MemoryFactory(const std::string& memoryName, std::shared_ptr<DeviceConfiguration>& deviceConfiguration);

		///
		/// Creates scalar memory of the given size
		/// \param size the number of real numbers to store
		/// \note size is in number of reals
		/// \returns the pointer to the newly created memory area.
		///
		std::shared_ptr<Memory<real> > createScalarMemory(size_t size);
	private:

		

		std::string memoryName;

		std::shared_ptr<DeviceConfiguration> deviceConfiguration;


    };
}
}
