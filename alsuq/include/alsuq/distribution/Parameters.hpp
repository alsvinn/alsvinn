#pragma once
#include "alsutils/parameters/Parameters.hpp"
#include "alsuq/types.hpp"
#include <string>
#include <map>
namespace alsuq { namespace distribution {

    class Parameters : public alsutils::parameters::Parameters {
    public:
        Parameters(const boost::property_tree::ptree &ptree);
        double getParameter(const std::string& name) const;

        void setParameter(const std::string& name, real value);


    private:
        std::map<std::string, real> parameters;
    };
} // namespace distribution
} // namespace alsuq
