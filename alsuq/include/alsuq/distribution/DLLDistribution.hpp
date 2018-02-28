#pragma once

#include <functional>
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"

namespace alsuq {
namespace distribution {

//! The DLL Distribution loads dynamically functions
//! from a user specified DLL File.
//!
//! This is ideal when you want to use some third party QMC
//! module that you don't want to include into the project
//!
//! The parameters you can supply are
//!
//! <table>
//! <tr><th> parameter name</th> <th>description</th></tr>
//! <tr><td>library          </td><td> filename of dll</td></tr>
//!
//! <tr><td>create_function  </td><td> name of the create function, should have the following signature<br />
//!                           \code{.cpp}
//!                              void* create_function(int size, int dimension, void* parameters);
//!                           \endcode
//!                           use NONE if it is not supplied</td></tr>
//!
//! <tr><td>delete_function </td><td> the function to delete any data created,
//!                           if create_function is NONE, this is ignored
//!                           assumes signature
//!                           \code{.cpp}
//!                              void delete_function(void* data);
//!                           \endcode</td></tr>
//!
//! <tr><td>generator_function </td><td> the name of the genreator function
//!                            assumes signature
//!                            \code{.cpp}
//!                               real generator_function(void* data, int size, int dimension, int component, int sample, void* parameters);
//!                            \endcode</td></tr>
//!
//! <tr><td>make_parameters_function </td><td> Name of the function to create the parameter struct
//!                            assumes the signature
//!                            \code{.cpp}
//!                               void* make_parameters_function();
//!                            \endcode</td></tr>
//!
//! <tr><td>delete_parameters_function </td><td> name of the function to delete the parameter struct
//!                            assumes the signature
//!                            \code{.cpp}
//!                               void delete_parameters_function(void* parameters);
//!                            \endcode</td></tr>
//!
//! <tr><td>set_parameters_function </td><td> set the parameter, assumes the signature
//!                            \code{.cpp}
//!                                void set_parameters_function(void* parameters, const char* key, const char* value);
//!                            \endcode</td></tr>
//! </table>
//!
class DLLDistribution : public Distribution {
public:
    typedef void* DLLData;
    DLLDistribution(size_t numberOfSamples, size_t dimension,
        const Parameters& parameters);
    virtual ~DLLDistribution();

    //! Generates the next number from the DLL
    virtual real generate(generator::Generator& generator, size_t component);


private:
    DLLData dllData = nullptr;
    std::function<real(DLLData, int, int, int, int, void*)> generatorFunction;
    std::function<void(DLLData)> deleteFunction;
    const int size = 0;
    const int dimension = 0;


    std::vector<int> samples;
    DLLData parametersStruct = nullptr;

    std::function < void(DLLData)> deleteParametersFunction;
};
} // namespace generator
} // namespace alsuq
