#include "alsfvm/io/Writer.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsfvm {
namespace io {

void Writer::addAttributes(const std::string& nameOfAttributes,
    const boost::property_tree::ptree& attributes) {
    if (attributesMap.find(nameOfAttributes) != attributesMap.end()) {
        THROW("Attribute " << nameOfAttributes << " already registered");
    }

    attributesMap[nameOfAttributes] = attributes;
}


} // namespace io
} // namespace alsfvm

