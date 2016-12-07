#include "alsuq/generator/Parameters.hpp"

namespace alsuq { namespace generator {

double Parameters::getParameter(std::string &name) const
{
    return parameters.at(name);
}

void Parameters::setParameter(const std::string &name, real value)
{
    if (parameters.find(name) != parameters.end()) {
        THROW("Parameter already registered: " << name);
    }
    parameters[name] = value;
}

}
}
