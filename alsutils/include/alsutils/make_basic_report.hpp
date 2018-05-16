#include <boost/property_tree/ptree.hpp>

namespace alsutils {

//! Writes a short boost::property_tree on the current alsvinn system
//!
//! This includes, but is not limited to: compiler version, CPU, GPU, MPI version, etc
//!
//! The output should *not* be used programatically, only for "human readable" output
//! afterwards to increase reproducability.
boost::property_tree::ptree makeBasicReport();
}
