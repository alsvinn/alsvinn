#pragma once

#include <functional>
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"

namespace alsuq {
namespace distribution {

//! The DLL Distribution loads a dynamically functions
//! from a user specified DLL File.
//!
//! This is ideal when you want to use some third party QMC
//! module that you don't want to include into the project
//!
//! The parameters you can supply are
//!
//!    parameter name        | description
//!    ----------------------|----------------------
//!     library              | filename of dll
//!                          |
//!     create_function      | name of the create function, should have the following signature
//!                          | \code{.cpp}
//!                          |    void* create_function(size, dimension);
//!                          | \endcode
//!                          | use NONE if it is not supplied
//!                          |
//!     delete_function      | the function to delete any data created,
//!                          | if create_function is NONE, this is ignored
//!                          | assumes signature
//!                          | \code{.cpp}
//!                          |    void delete_function(void* data);
//!                          | \endcode
//!                          |
//!     generator_function   | the name of the genreator function
//!                          | assumes signature
//!                          | \code{.cpp}
//!                          |    real generator_function(void* data, int size, int dimension, int component, int sample);
//!                          | \endcode
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
        std::function<real(DLLData, int, int, int, int)> generatorFunction;
        std::function<void(DLLData)> deleteFunction;
        const int size = 0;
        const int dimension = 0;


        std::vector<int> samples;
};
} // namespace generator
} // namespace alsuq
