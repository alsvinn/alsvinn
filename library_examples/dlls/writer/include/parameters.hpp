#pragma once
#include <string>
#include <map>
#include <mpi.h>

//! Simple example parameter class that can be used
class MyParameters {
public:

    void setParameter(const std::string& key, const std::string& value) {
        parameters[key] = value;
    }

    std::string getParameter(const std::string& key) const {
        return parameters.at(key);
    }

    void setMPIComm(MPI_Comm comm) {
        mpiComm = comm;
    }


    MPI_Comm getMPIComm(MPI_Comm comm) {
        return comm;
    }

private:
    std::map<std::string, std::string> parameters;

    MPI_Comm mpiComm;

};
