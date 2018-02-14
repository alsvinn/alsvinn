#include "alsfvm/config/GenericParameters.hpp"

namespace alsfvm {
namespace config {

GenericParameters::GenericParameters(const boost::property_tree::ptree& tree)
    : tree(tree) {

}

double GenericParameters::getDouble(const std::string& key) const {
    return tree.get<double>(key);
}

int GenericParameters::getInteger(const std::string& key) const {
    return tree.get<int>(key);
}

double GenericParameters::getDouble(const std::string& key,
    double defaultValue) const {
    if (tree.find(key) != tree.not_found()) {
        return getDouble(key);
    } else {
        return defaultValue;
    }
}

int GenericParameters::getInteger(const std::string& key,
    int defaultValue) const {
    if (tree.find(key) != tree.not_found()) {
        return getInteger(key);
    } else {
        return defaultValue;
    }
}

}
}
