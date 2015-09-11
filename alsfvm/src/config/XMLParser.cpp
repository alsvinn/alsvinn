#include "alsfvm/config/XMLParser.hpp"
#include <boost/property_tree/xml_parser.hpp>

namespace alsfvm { namespace config {

void XMLParser::parseFile(std::ifstream &stream, ptree& configuration)
{
    boost::property_tree::read_xml(stream, configuration);
}

}
}
