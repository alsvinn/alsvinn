#pragma once
#include "alsfvm/mpi/Request.hpp"

namespace alsfvm {
namespace mpi {

//! Holds a collection of requests
class RequestContainer {
public:
    typedef Request::RequestPtr RequestPtr;

    void addRequest(RequestPtr request);

    void waitForAll();

private:
    std::vector<RequestPtr> requests;
};
} // namespace mpi
} // namespace alsfvm
