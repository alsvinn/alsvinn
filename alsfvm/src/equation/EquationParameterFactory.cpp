#include "alsfvm/equation/EquationParameterFactory.hpp"
#include "alsfvm/equation/euler/EulerParameters.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm { namespace equation {

namespace {
struct EquationParametersFunctor {
    EquationParametersFunctor(const std::string& name,
                              alsfvm::shared_ptr<EquationParameters>& parameters)
        : name(name), parameters(parameters)
    {

    }

    template<class T>
    void operator()(const T& t) const {
        if (T::getName() == name) {
            parameters.reset(new typename T::EquationType::Parameters);
        }
    }

    std::string name;
    alsfvm::shared_ptr<EquationParameters>& parameters;
};
}

alsfvm::shared_ptr<EquationParameters> EquationParameterFactory::createDefaultEquationParameters(const std::string &name)
{
    alsfvm::shared_ptr<EquationParameters> parameters;
    EquationParametersFunctor equationParametersFunctor(name, parameters);


    for_each_equation(equationParametersFunctor);

    if (!parameters) {
        THROW("Unknown equation " << name);
    }

    return parameters;
}

}
}
