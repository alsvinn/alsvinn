#pragma once

#define MPI_SAFE_CALL(X) {  \
    int error = X; \
    if (error != MPI_SUCCESS) { \
            std::cerr << "Noticed MPI Error in " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "\tLine was:\"" << #X << "\"" << std::endl; \
            THROW("MPI error" << std::endl << "Line was: " << std::endl <<"\t" << #X << "\nError code: " << error); \
    } \
}
