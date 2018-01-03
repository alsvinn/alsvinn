#include "alsfvm/simulator/TimestepInformation.hpp"

namespace alsfvm {
namespace simulator {

TimestepInformation::TimestepInformation(real currentTime,
                                         size_t numberOfStepsPerformed)
    : currentTime(currentTime), numberOfStepsPerformed(numberOfStepsPerformed)
{

}

TimestepInformation::TimestepInformation()
    : currentTime(0), numberOfStepsPerformed(0)
{

}

void TimestepInformation::incrementTime(real dt)
{
    currentTime += dt;
    numberOfStepsPerformed++;
}

real TimestepInformation::getCurrentTime() const
{
    return currentTime;
}

size_t TimestepInformation::getNumberOfStepsPerformed() const
{
    return numberOfStepsPerformed;
}

} // namespace simulator
} // namespace alsfvm

