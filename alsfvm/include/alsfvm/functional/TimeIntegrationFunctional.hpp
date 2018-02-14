#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/functional/Functional.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"

namespace alsfvm {
namespace functional {

//! This lets you time integrate a functional, that is, for a functional
//! g (interpreted in the loose sense), this will compute
//!
//! \f[\int_{t-\tau}^{t+\tau} g(u(t))\; dt\f]
//!
//! @note this computes the time integral *without* averaging,
//!       if you want to get the time averaged quantity, you have to divide
//!       the output by \f$2\tau\f$.
//!
//! @note It is not really easy to combine this into the time integration class
//!       for writing. The reason for this is that we only selectively want to call
//!       the functional, to minimize computational work.
class TimeIntegrationFunctional : public io::Writer {
public:

    TimeIntegrationFunctional(volume::VolumeFactory volumeFactory,
        io::WriterPointer writer,
        FunctionalPointer functional,
        double time,
        double timeRadius);

    virtual void write(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

    virtual void finalize(const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

private:
    void makeVolumes(const grid::Grid& grid);
    volume::VolumeFactory volumeFactory;
    io::WriterPointer writer;
    FunctionalPointer functional;

    volume::VolumePointer conservedVolume;
    volume::VolumePointer extraVolume;

    const double time;
    const double timeRadius;
    double lastTime = 0;

    ivec3 functionalSize;

};
} // namespace functional
} // namespace alsfvm
