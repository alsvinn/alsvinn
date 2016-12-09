#pragma once
#include "alsuq/types.hpp"
#include <string>
#include <map>
namespace alsuq { namespace distribution {

    class Parameters {
    public:
        double getParameter(const std::string& name) const;

        void setParameter(const std::string& name, real value);

    private:
        std::map<std::string, real> parameters;
    };
} // namespace distribution
} // namespace alsuq
