#pragma once
#include <boost/property_tree/ptree.hpp>
namespace alsutils { namespace parameters { 

    //! Holds general parameters based on a boost::property tree
    class Parameters {
    public:
        Parameters(const boost::property_tree::ptree& ptree);
        double getDouble(const std::string& name) const;
        int getInteger(const std::string& name) const;
        std::string getString(const std::string& name) const;


    private:
        boost::property_tree::ptree ptree;

    };
} // namespace parameters
} // namespace alsutils
