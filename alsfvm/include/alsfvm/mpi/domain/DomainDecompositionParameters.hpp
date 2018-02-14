#pragma once
#include "alsfvm/config/GenericParameters.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {

class DomainDecompositionParameters
    : public config::GenericParameters {
    public:
        DomainDecompositionParameters(const boost::property_tree::ptree& ptree);
};
} // namespace domain
} // namespace mpi
} // namespace alsfvm
