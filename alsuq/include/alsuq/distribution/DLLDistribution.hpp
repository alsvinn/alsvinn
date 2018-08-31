/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <functional>
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"

namespace alsuq {
namespace distribution {

//! The DLL Distribution loads functions dynamically
//! from a user specified DLL file (shared library) at run time.
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
//! <tr><td>set_parameter_function </td><td> set the parameter, assumes the signature
//!                            \code{.cpp}
//!                                void set_parameter_function(void* parameters, const char* key, const char* value);
//!                            \endcode</td></tr>
//! </table>
//!
//! set_parameter_function will be called for every parameter given to the distribution
//! tag, ie if the xml given was
//!
//! \code{.xml}
//! <distribution>
//!    <type>dll</type>
//!    <library>libfoo.so</library>
//!    ...
//!    <optionA>value</optionA>
//! </distribution>
//! \endcode
//!
//! the following code will essentially be called
//!
//! \code{.cpp}
//! parameters = make_parameters_function();
//! set_parameter_function(parameters, "type", "dll");
//! set_parameter_function(parameters, "library", "libfoo.so");
//! // ...
//! set_parameter_function(parameters, "optionA", "value");
//!
//! // In addition, we always supply the mpi_node and mpi_size
//! // (from MPI_Comm_rank and MPI_Comm_size)
//! set_parameter_function(parameters, "mpi_node", mpi_node_as_string);
//! set_parameter_function(parameters, "mpi_size", mpi_size_as_string);
//! \endcode
//!
//! To get an idea on how this is called, this is a rough sketch (the code
//! assumes every option is set, there are some if tests to check that the user
//! supplied every function, otherwise they are gracefully skipped):
//!
//! \code{.cpp}
//! void init() {
//!     parameters = make_parameters_function();
//!     set_parameter_function(parameters, "type", "dll");
//!     set_parameter_function(parameters, "library", "libfoo.so");
//!     // ...
//!     set_parameter_function(parameters, "optionA", "value");
//!
//!     // In addition, we always supply the mpi_node and mpi_size
//!     // (from MPI_Comm_rank and MPI_Comm_size)
//!     set_parameter_function(parameters, "mpi_node", mpi_node_as_string);
//!     set_parameter_function(parameters, "mpi_size", mpi_size_as_string);
//!
//!     data = create_function(size, dimension, parameters);
//!
//!     // you can assume mpi gets synchronized at this point, that is, we
//!     // will essentially call
//!     MPI_Barrier(MPI_COMM_WORLD);
//! }
//!
//! double generate_sample(int component, int sample) {
//!     return generate_function(data, size, dimension, component, sample, parameters);
//! }
//!
//! void destruct() {
//!     // first delete the data
//!     delete_function(data);
//!     // then delete the parameters
//!     delete_parameters_function(parameters);
//! }
//! \endcode
//!
class DLLDistribution : public Distribution {
public:
    typedef void* DLLData;
    DLLDistribution(size_t numberOfSamples, size_t dimension,
        const Parameters& parameters);
    virtual ~DLLDistribution();

    //! Generates the next number from the DLL
    virtual real generate(generator::Generator& generator, size_t component,
        size_t sample) override;


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
