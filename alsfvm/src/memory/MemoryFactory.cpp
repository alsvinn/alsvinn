#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/error/Exception.hpp"


namespace alsfvm {
	namespace memory {
		namespace {
			///
			/// Singleton pattern
			///
			std::map < std::string, MemoryFactory::MemoryConstructor >& getMemoryFactoryMapInstance() {
				// The order of initialization is undefined, therefore we must use a singleton for this map
				// since the static variables within functions are always initialized by the first function call
				static std::map < std::string, MemoryFactory::MemoryConstructor > memoryFactoryMap;
				return memoryFactoryMap;
			}
		}

		void MemoryFactory::addConstructor(const std::string& name, MemoryConstructor constructor) {
			auto& constructors = getMemoryFactoryMapInstance();
			constructors[name] = constructor;
		}

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
			auto& constructors = getMemoryFactoryMapInstance();
			if (constructors.find(memoryName) == constructors.end()) {
                THROW("Unrecognized memory name" << memoryName);
			}
		
            return std::dynamic_pointer_cast<Memory<real> > (constructors[memoryName](nx, ny, nz, REAL, deviceConfiguration));

		}
	}
}
