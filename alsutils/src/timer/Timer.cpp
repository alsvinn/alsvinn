#include "alsutils/timer/Timer.hpp"

namespace alsutils {
namespace timer {

Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>
        (end - start).count();

    data.addTime(duration);

}

}
}
