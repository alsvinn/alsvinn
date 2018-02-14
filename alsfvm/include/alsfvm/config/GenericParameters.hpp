#pragma once
#include <boost/property_tree/ptree.hpp>
namespace alsfvm {
namespace config {

//! Base class to hold parameters through a boost property tree. To be
//! passed around to other classes.
class GenericParameters {
    public:
        GenericParameters(const boost::property_tree::ptree& tree);

        //! Gets a double parameter with key key
        double getDouble(const std::string& key) const;

        //! Gets an integer parameter with key key
        int getInteger(const std::string& key) const;


        //! Gets a double parameter with key key,
        //!
        //! if they key does not exist, returns defaultValue
        double getDouble(const std::string& key, double defaultValue) const;

        //! Gets an integer parameter with key key
        //!
        //! if they key does not exist, returns defaultValue
        int getInteger(const std::string& key, int defaultValue) const;
    private:
        boost::property_tree::ptree tree;
};
} // namespace config
} // namespace alsfvm
