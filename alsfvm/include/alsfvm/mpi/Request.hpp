#pragma once
#include <mpi.h>
#include "alsfvm/mpi/Configuration.hpp"
#include "alsfvm/types.hpp"
#include <memory>
#ifdef ALSVINN_HAVE_CUDA
    #include <thrust/host_vector.h>
#endif

namespace alsfvm {
namespace mpi {

//! Holds request information
//!
//! \note Should be held in a unique_ptr
class Request {
public:
    //! Singleton
    Request();
public:
    typedef alsfvm::shared_ptr<Request> RequestPtr;

    //! Maps to MPI_Isend. See http://www.mpich.org/static/docs/v3.1/www3/MPI_Isend.html
    template<class Data>
    static RequestPtr isend(const Data& data, int count, MPI_Datatype datatype,
        int destination, int tag, Configuration& configuration);

    //! Maps to MPI_Irecv. See http://www.mpich.org/static/docs/v3.1/www3/MPI_Irecv.html
    template<class Data>
    static RequestPtr ireceive(Data& receiveBuffer, int count,
        MPI_Datatype datatype,
        int source, int tag, Configuration configuration);


    //! Wait for the request to finish, maps to MPI_Wait.
    void wait();

    friend class std::unique_ptr<Request>;

    ~Request();
private:

    MPI_Request request{NULL};
};

typedef Request::RequestPtr RequestPtr;

template<class Data>
inline  RequestPtr Request::isend(const Data& data, int count,
    MPI_Datatype datatype,
    int destination, int tag, Configuration& configuration) {
    std::shared_ptr<Request> requestPointer(new Request());


    MPI_Isend((void*)data.getPointer(), count, datatype, destination, tag,
        configuration.getCommunicator(),
        &requestPointer->request);

    return requestPointer;
}

template<class Data>
inline  RequestPtr Request::ireceive(Data& receiveBuffer, int count,
    MPI_Datatype datatype,
    int source, int tag, Configuration configuration) {
    std::shared_ptr<Request> requestPointer(new Request());


    MPI_Irecv((void*)receiveBuffer.getPointer(), count, datatype, source, tag,
        configuration.getCommunicator(),
        &requestPointer->request);

    return requestPointer;
}


#ifdef ALSVINN_HAVE_CUDA
template<>
inline  RequestPtr Request::isend(const thrust::host_vector<real>& data,
    int count, MPI_Datatype datatype,
    int destination, int tag, Configuration& configuration) {
    std::shared_ptr<Request> requestPointer(new Request());


    MPI_Isend((void*)data.data(), count, datatype, destination, tag,
        configuration.getCommunicator(),
        &requestPointer->request);

    return requestPointer;
}

template<>
inline  RequestPtr Request::ireceive(thrust::host_vector<real>& receiveBuffer,
    int count, MPI_Datatype datatype,
    int source, int tag, Configuration configuration) {
    std::shared_ptr<Request> requestPointer(new Request());


    MPI_Irecv((void*)receiveBuffer.data(), count, datatype, source, tag,
        configuration.getCommunicator(),
        &requestPointer->request);

    return requestPointer;
}
#endif
}
} // namespace alsfvm
