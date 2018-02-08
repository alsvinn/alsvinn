#include "alsuq/distribution/Parameters.hpp"
#include "alsutils/error/Exception.hpp"
namespace alsuq { namespace distribution {

Parameters::Parameters(const boost::property_tree::ptree &ptree)
    : alsutils::parameters::Parameters(ptree)
{

}

double Parameters::getParameter(const std::string &name) const
{
    if (parameters.find(name) == parameters.end()) {
        THROW("Unknown parameter " << name);
    }
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
