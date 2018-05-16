#pragma once
#include <boost/property_tree/ptree.hpp>
#include <boost/version.hpp>
#include <boost/config.hpp>
namespace alsutils {
inline boost::property_tree::ptree getBoostProperties() {
    boost::property_tree::ptree properties;
    properties.put("BOOST_VERSION", BOOST_VERSION);
    properties.put("BOOST_LIB_VERSION", BOOST_LIB_VERSION);
    properties.put("BOOST_PLATFORM", BOOST_PLATFORM);
    properties.put("BOOST_PLATFORM_CONFIG", BOOST_PLATFORM_CONFIG);
    properties.put("BOOST_COMPILER", BOOST_COMPILER);
    #ifdef BOOST_LIBSTDCXX_VERSION
    properties.put("BOOST_LIBSTDCXX_VERSION", BOOST_LIBSTDCXX_VERSION);
    #endif
    #ifdef BOOST_LIBSTDCXX11
    properties.put("C++11", true);
#else
    properties.add("C++11", false);
    #endif
    properties.put("BOOST_STDLIB", BOOST_STDLIB);
    properties.put("BOOST_STDLIB_CONFIG", BOOST_STDLIB_CONFIG);

    return properties;
}
}
