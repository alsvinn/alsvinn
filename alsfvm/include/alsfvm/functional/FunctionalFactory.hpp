#pragma once
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm { namespace functional { 

    //! Factory class for creating a functional
    class FunctionalFactory {
    public:
        typedef Functional::Parameters Parameters;
        typedef std::function<FunctionalPointer(const Parameters&)> FunctionalCreator;

        static void registerFunctional(const std::string& platform,
                                       const std::string &name,
                                       FunctionalFactory::FunctionalCreator maker);

        FunctionalPointer makeFunctional(const std::string &platform,
                                                            const std::string &name,
                                                            const FunctionalFactory::Parameters &parameters);
    };
} // namespace functional
} // namespace alsfvm
