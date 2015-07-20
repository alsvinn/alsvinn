#pragma once
///
/// This file includes various utility functions for the numerical fluxes
///

#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include <iostream>
///
/// Macro to add class to the numerical flux factory
/// \note classname should be classname without template or namespace (must be called within namespace)
///
#define ADD_NUMERICALFLUX_TO_FACTORY(platform, equation, fluxname, classname) \
    namespace { /* so that we do not export the class */ \
        struct InitClass {/* See http://stackoverflow.com/a/10897578  for explanation on how this works*/ \
                InitClass() { \
                    std::cout << "here2" << std::endl;\
                    alsfvm::numflux::NumericalFluxFactory::addConstructor(platform, equation, #fluxname, [=](const grid::Grid& grid, \
                            const std::string& reconstruction,\
                            std::shared_ptr<DeviceConfiguration>& deviceConfiguration) { \
                       if (reconstruction == "none") { \
                            alsfvm::numflux::NumericalFluxFactory::NumericalFluxPtr ptr(new classname<fluxname>(grid, deviceConfiguration)); \
                                                return ptr; \
                                        } else { \
                        THROW("Could not find reconstruction type"); \
                                        } \
                                }); \
                        } \
                } globalVariableForInitialization;  /*global variable, constructor will be run before main()*/\
        }


