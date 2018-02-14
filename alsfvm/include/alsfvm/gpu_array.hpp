#pragma once

namespace alsfvm {

///
/// \brief The gpu_array class is akin to the std::array, only also works for
///        gpus
///
template<class T, size_t N>
class gpu_array {
public:

    __host__ __device__ gpu_array() {
        // empty
    }

    __host__ __device__ gpu_array(std::initializer_list<T> initializerList) {
        int i = 0;

        for (const T& t : initializerList) {
            data[i++] = t;
        }
    }

    __host__ __device__ T& operator[](int i) {
        return data[i];
    }

    __host__ __device__ const T& operator[](int i) const {
        return data[i];
    }

    size_t size() const {
        return N;
    }
private:
    T data[N];

};



}
