#pragma once
#include "alsfvm/io/FixedIntervalWriter.hpp"
#include "alsfvm/functional/Functional.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"

namespace alsfvm { namespace functional {


///
/// \brief The IntervalFunctionalWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed time intervals.
///
/// This class is useful if you only want to save every x seconds of simulation. This class assume you
/// already decorates it with the alsfvm::io::FixedIntervalWriter
///
    class IntervalFunctionalWriter : public io::Writer {
    public:

        IntervalFunctionalWriter(volume::VolumeFactory volumeFactory,
                                  io::WriterPointer writer,
                                  FunctionalPointer functional
                                  );

        virtual void write(const volume::Volume& conservedVariables,
                           const volume::Volume& extraVariables,
                           const grid::Grid& grid,
                           const simulator::TimestepInformation& timestepInformation) override;


    private:
        void makeVolumes(const grid::Grid& grid);
        volume::VolumeFactory volumeFactory;
        io::WriterPointer writer;
        FunctionalPointer functional;

        volume::VolumePointer conservedVolume;
        volume::VolumePointer extraVolume;



        ivec3 functionalSize;

    };
} // namespace functional
} // namespace alsfvm
