#pragma once
#include "alsfvm/types.hpp"

///
/// Utility file for the memory components, this is not meant to be used
/// from other parts of the system.
///

///
/// Template instatiates the class for the different types we need
///
#define INSTANTIATE_MEMORY(classname) \
    template class classname<alsfvm::real>;

