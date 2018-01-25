#pragma once
#include <boost/property_tree/ptree.hpp>
#include <map>
namespace alsutils { namespace parameters { 

    //! Holds general parameters based on a boost::property tree
    class Parameters {
    public:
        Parameters(const boost::property_tree::ptree& ptree);

        //! Convenience constructor. Used mostly for unittesting.
        //!
        Parameters(const std::map<std::string, std::string>& values);


        double getDouble(const std::string& name) const;
        int getInteger(const std::string& name) const;
        std::string getString(const std::string& name) const;


    private:
        boost::property_tree::ptree ptree;

    };
} // namespace parameters
} // namespace alsutils
