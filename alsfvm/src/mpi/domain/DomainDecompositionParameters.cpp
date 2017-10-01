#include "alsfvm/mpi/domain/DomainDecompositionParameters.hpp"

namespace alsfvm { namespace mpi { namespace domain {

DomainDecompositionParameters::DomainDecompositionParameters(const boost::property_tree::ptree &ptree)
    : config::GenericParameters(ptree)
{

    // empty
}

}
}
}
