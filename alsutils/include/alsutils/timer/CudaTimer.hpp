#pragma once
#include "alsutils/timer/TimerDatabase.hpp"
#include "alsutils/timer/Timer.hpp"
#include "alsutils/config.hpp"
#include <cuda_runtime.h>
#include <chrono>
namespace alsutils {
namespace timer {

class CudaTimer {
public:
    template<class ...T> CudaTimer(cudaStream_t stream, T... names) :
        data(TimerDatabase::getInstance().getTimerData(names...)),
        stream(stream),
        start(new
            std::chrono::high_resolution_clock::time_point()) {

        addStartCallback();
    }

    ~CudaTimer() noexcept(false);

private:

    TimerData& data;
    cudaStream_t stream;
    void addStartCallback();

    // This is not the cleanest way of doing it, but at the moment I see no other
    // way: We first allocate the space for storing the starting time on the heap
    // this is then deleted in the second callback.
    std::chrono::high_resolution_clock::time_point* start;

};
} // namespace timer
} // namespace alsutils
#ifdef ALSVINN_USE_CUDA_TIMERS
    #define ALSVINN_TIME_CUDA_BLOCK(STREAM, ...) ::alsutils::timer::CudaTimer ALSVINN_MAKE_TIMER_VARIABLE_NAME(__VA_ARGS__) (STREAM, ALSVINN_MAKE_TIMER_STRINGS(__VA_ARGS__))
#else
    #define ALSVINN_TIME_CUDA_BLOCK(STREAM, ...)
#endif
