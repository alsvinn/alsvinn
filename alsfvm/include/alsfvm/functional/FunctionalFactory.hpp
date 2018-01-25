#pragma once
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm { namespace functional { 

    //! Factory class for creating a functional
    class FunctionalFactory {
    public:
        typedef Functional::Parameters Parameters;

        std::map<std::string,

        //! Create functional
        alsfvm::shared_ptr<Functional> createFunctional(const std::string& name,
                                                        const Parameters& parameters);
    };
} // namespace functional
} // namespace alsfvm
