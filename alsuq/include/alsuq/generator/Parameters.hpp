#pragma once
#include "alsutils/types.hpp"
#include <string>
#include <map>
namespace alsuq { namespace generator { 

    class Parameters {
    public:
        double getParameter(std::string& name) const;

        void setParameter(const std::string& name, real value);

    private:
        std::map<std::string, real> parameters;
    };
} // namespace generator
} // namespace alsuq
