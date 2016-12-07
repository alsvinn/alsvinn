#pragma once
#include <string>
#include <map>
namespace alsuq { namespace generator { 

    class Parameters {
    public:
        double getParameter(std::string& name) const;

        void setParameter(const std::string& name);

    private:
        std::map<std::string, double> parameters;
    };
} // namespace generator
} // namespace alsuq
