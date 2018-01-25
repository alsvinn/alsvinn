#include "alsutils/parameters/Parameters.hpp"

namespace alsutils { namespace parameters {

Parameters::Parameters(const boost::property_tree::ptree &ptree)
    : ptree(ptree)
{

}

Parameters::Parameters(const std::map<std::string, std::string> &values)
{
    for(const auto& p : values) {
        ptree.add(p.first, p.second);
    }
}

double Parameters::getDouble(const std::string &name) const
{
    return ptree.get<double>(name);
}

int Parameters::getInteger(const std::string &name) const
{
   return ptree.get<int>(name);
}

std::string Parameters::getString(const std::string &name) const
{
    return ptree.get<std::string>(name);
}

}
}
