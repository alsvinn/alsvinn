#pragma once
#include <chrono>
#include "alsutils/timer/TimerData.hpp"
#include "alsutils/timer/TimerDatabase.hpp"
#include "alsutils/config.hpp"
namespace alsutils {
namespace timer {

class Timer {
public:
    template<class ...T> Timer(T... names)
        : start(std::chrono::high_resolution_clock::now()),
          data(TimerDatabase::getInstance().getTimerData(names...)) {

    }

    ~Timer();

private:
    const std::chrono::high_resolution_clock::time_point start;
    TimerData& data;
};
} // namespace timer
} // namespace alsutils



#define ALSVINN_MAKE_TIMER_VARIABLE_NAME1(X) timer##X
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME2(X, Y) timer##X##Y
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME3(X,Y, Z) timer##X##Y##Z
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME4(X,Y, Z,V) timer##X##Y##Z##V
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME5(X,Y, Z,V,W) timer##X##Y##Z##V##W
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME6(X,Y, Z,V,W,Q) timer##X##Y##Z##V##W##Q

#define ALSVINN_MAKE_TIMER_STRINGS1(X) #X
#define ALSVINN_MAKE_TIMER_STRINGS2(X, Y) #X,#Y
#define ALSVINN_MAKE_TIMER_STRINGS3(X,Y, Z) #X,#Y,#Z
#define ALSVINN_MAKE_TIMER_STRINGS4(X,Y, Z,V) #X, #Y, #Z,#V
#define ALSVINN_MAKE_TIMER_STRINGS5(X,Y, Z,V,W) #X, #Y, #Z, #V, #W
#define ALSVINN_MAKE_TIMER_STRINGS6(X,Y, Z,V,W, Q) #X, #Y, $Z, #V, #W, #Q

// see https://stackoverflow.com/a/11763277 for explanation of GET_MACRO
#define GET_MACRO(_1,_2,_3, _4, _5, _6, NAME,...) NAME
#define ALSVINN_MAKE_TIMER_VARIABLE_NAME(...) GET_MACRO(__VA_ARGS__, ALSVINN_MAKE_TIMER_VARIABLE_NAME6, ALSVINN_MAKE_TIMER_VARIABLE_NAME5, ALSVINN_MAKE_TIMER_VARIABLE_NAME4, ALSVINN_MAKE_TIMER_VARIABLE_NAME3, ALSVINN_MAKE_TIMER_VARIABLE_NAME2, ALSVINN_MAKE_TIMER_VARIABLE_NAME1)(__VA_ARGS__)
#define ALSVINN_MAKE_TIMER_STRINGS(...) GET_MACRO(__VA_ARGS__, ALSVINN_MAKE_TIMER_VARIABLE_NAME6, ALSVINN_MAKE_TIMER_STRINGS5, ALSVINN_MAKE_TIMER_STRINGS4,ALSVINN_MAKE_TIMER_STRINGS3, ALSVINN_MAKE_TIMER_STRINGS2, ALSVINN_MAKE_TIMER_STRINGS1)(__VA_ARGS__)

#ifdef ALSVINN_USE_TIMERS
    #define ALSVINN_TIME_BLOCK(...) ::alsutils::timer::Timer ALSVINN_MAKE_TIMER_VARIABLE_NAME(__VA_ARGS__) (ALSVINN_MAKE_TIMER_STRINGS(__VA_ARGS__))
#else
    #define ALSVINN_TIME_BLOCK(...)
#endif
