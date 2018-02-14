#pragma once
#include "alsfvm/types.hpp"
#include <iostream>
#include <boost/property_tree/ptree.hpp>

namespace alsfvm {
namespace config {

class XMLParser {
public:
    typedef boost::property_tree::ptree ptree;

    ///
    /// \brief parseFile parses the file and reads all the properties into
    /// the property tree
    /// \param[in] stream the stream to read from
    /// \param[out] configuration the configuration to write to.
    ///
    void parseFile(std::ifstream& stream, ptree& configuration);
};
} // namespace alsfvm
} // namespace config
