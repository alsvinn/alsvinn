#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace simulator {

class TimestepInformation {
public:
    TimestepInformation(real currentTime, size_t numberOfStepsPerformed);
    TimestepInformation();

    ///
    /// \brief incrementTime increments the current simulation time
    /// \param dt the increment size.
    ///
    void incrementTime(real dt);

    ///
    /// \return the current simulation time
    ///
    real getCurrentTime() const;

    ///
    /// \brief getNumberOfStepsPerformed returns the number of timesteps calculated.
    ///
    size_t getNumberOfStepsPerformed() const;

private:
    real currentTime;
    size_t numberOfStepsPerformed;
};

} // namespace simulator
} // namespace alsfvm


