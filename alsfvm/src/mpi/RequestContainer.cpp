#include "alsfvm/mpi/RequestContainer.hpp"

namespace alsfvm {
namespace mpi {

void RequestContainer::addRequest(RequestPtr request) {
    requests.push_back(request);
}

void RequestContainer::waitForAll() {
    for (auto& request : requests) {
        request->wait();
    }
}

}
}
