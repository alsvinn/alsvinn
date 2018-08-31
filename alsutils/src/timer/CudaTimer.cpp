#include "alsutils/timer/CudaTimer.hpp"
#include "alsutils/cuda/cuda_safe_call.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace alsutils {
namespace timer {

void CudaTimer::addStartCallback() {

    // note that we can not take reference since it is a function pointer, see eg
    // https://stackoverflow.com/questions/28746744/passing-lambda-as-function-pointer

    // See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
    // for how callbacks work
    auto callback = [](cudaStream_t, cudaError_t, void* startAsVoid) -> void {
        auto startConverted = reinterpret_cast<std::chrono::high_resolution_clock::time_point*>(startAsVoid);
        * startConverted = std::chrono::high_resolution_clock::now();
    };

    CUDA_SAFE_CALL(cudaStreamAddCallback(stream, callback, start, 0));
}

CudaTimer::~CudaTimer() noexcept(false) {
    // This is rather dirty, but since this object gets deleted after, we need to
    // allocate dynamic memory like this

    auto dataToSend = new
    std::pair<TimerData&, std::chrono::high_resolution_clock::time_point*>(data,
        start);


    // See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
    auto callback = [](cudaStream_t, cudaError_t, void* dataAsVoid) {
        auto dataConverted =
            reinterpret_cast<std::pair<TimerData&, std::chrono::high_resolution_clock::time_point*>*>
            (dataAsVoid);

        auto end = std::chrono::high_resolution_clock::now();

        auto start = dataConverted->second;
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>
            (end - *start).count();
        dataConverted->first.addTime(duration);

        delete start;
        delete dataConverted;
    };

    CUDA_SAFE_CALL(cudaStreamAddCallback(stream, callback, dataToSend, 0));
}

}
}
