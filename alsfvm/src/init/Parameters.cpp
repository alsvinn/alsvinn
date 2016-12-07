#include "alsfvm/init/Parameters.hpp"
#include "alsfvm/error/Exception.hpp"
namespace alsfvm { namespace init { 
    //! Add a a parameter to the parameters.
    void Parameters::addParameter(const std::string& name, const std::vector<real>& value) {
        if (parameters.find(name) != parameters.end()) {
            THROW("Parameter already registered: " << name);
        }

        parameters[name] = value;
    }


    std::vector<std::string> Parameters::getParameterNames() const {
        std::vector<std::string> names;
        for (auto key : parameters) {
            names.push_back(key.first);
        }

        return names;
    }

    //! Each parameter is represented by an array 
    //! A scalar is then represented by a length one array.
    std::vector<double> Parameters::getParameter(const std::string& name) const {
        if (parameters.find(name) == parameters.end()) {
            THROW("Parameter not found: " << name);
        }

        return parameters.at(name);
    }
}
}
