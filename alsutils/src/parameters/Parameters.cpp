#include "alsutils/parameters/Parameters.hpp"
#include <boost/algorithm/string.hpp>
namespace alsutils {
namespace parameters {

Parameters::Parameters(const boost::property_tree::ptree& ptree)
    : ptree(ptree) {

}

Parameters::Parameters(const std::map<std::string, std::string>& values) {
    for (const auto& p : values) {
        ptree.add(p.first, p.second);
    }
}

double Parameters::getDouble(const std::string& name) const {
    return ptree.get<double>(name);
}

int Parameters::getInteger(const std::string& name) const {
    return ptree.get<int>(name);
}

std::string Parameters::getString(const std::string& name) const {
    return ptree.get<std::string>(name);
}

bool Parameters::contains(const std::string& name) const {
    return ptree.find(name) != ptree.not_found();
}

std::vector<std::string> Parameters::getStringVectorFromString(
    const std::string& name) const {
    std::vector<std::string> strings;
    std::string inputString = ptree.get<std::string>(name);

    boost::split(strings, inputString, boost::is_any_of(" \t"));

    return strings;
}

}
}
