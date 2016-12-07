#pragma once
#include "alsuq/generator/Generator.hpp"
#include "alsuq/generator/Parameters.hpp"

namespace alsuq { namespace generator { 

    class GeneratorFactory {
    public:

        //!
        //! \brief makeGenerator creates a new generator
        //! \param name the name of the generator
        //! \param dimensions the number of dimensions to use
        //! \param numberVariables number of random variables to draw (relevant for QMC)
        //! \param parameters additional paramters
        //! \return the new generator
        //!
        std::shared_ptr<Generator> makeGenerator(const std::string& name,
                                                 const size_t dimensions,
                                                 const size_t numberVariables,
                                                 const Parameters& parameters);


    };
} // namespace generator
} // namespace alsuq
