#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/error/Exception.hpp"
///
/// Utility file for the memory components, this is not meant to be used
/// from other parts of the system.
///


///
/// Macro to add class to the memory factory
/// \note classname should be classname without template or namespace (must be called within namespace)
///
#define ADD_MEMORY_TO_FACTORY(classname) \
	namespace { /* so that we do not export the class */ \
	struct InitClass {/* See http://stackoverflow.com/a/10897578  for explanation on how this works*/ \
		InitClass() { \
            alsfvm::memory::MemoryFactory::addConstructor(#classname, [](size_t nx, size_t ny, size_t nz, Types type,  \
			std::shared_ptr<DeviceConfiguration>& deviceConfiguration) { \
					if (type == alsfvm::REAL) { \
                        alsfvm::memory::MemoryFactory::MemoryPtr ptr(new classname<alsfvm::real>(nx, ny, nz)); \
						return ptr; \
					} else { \
                        THROW("Could not find memory type"); \
					} \
				}); \
			} \
		} globalVariableForInitialization;  /*global variable, constructor will be run before main()*/\
	}


///
/// Template instatiates the class for the different types we need
///
#define INSTANTIATE_MEMORY(classname) \
    template class classname<alsfvm::real>;


