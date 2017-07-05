// Test if the volume view works correctly.
// The volume view is essentially the 
// second constructor of the volume 
// that takes another volume along with
// a  selection of variables to view
#include <gtest/gtest.h>
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"

TEST(VolumeViewTest, AccessVolumeTest) {
    const std::string equation = "euler3";
    size_t N = 5;
    auto deviceConfiguration = alsfvm::make_shared<alsfvm::DeviceConfiguration>("cpu");
    auto memoryFactory = alsfvm::make_shared<alsfvm::memory::MemoryFactory>(deviceConfiguration);
    auto volumeFactory = alsfvm::make_shared<alsfvm::volume::VolumeFactory>(equation, memoryFactory);

    auto volume = volumeFactory->createConservedVolume(N, 1, 1);

    for (size_t i = 0; i < N; ++i) {
        for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
            volume->getScalarMemoryArea(var)->getPointer()[i] = i*N + var;
        }
    }

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        alsfvm::volume::Volume view(*volume, { var }, { "u" });

        for (size_t i = 0; i < N; ++i) {
            ASSERT_EQ(view[0]->getPointer()[i], i*N + var);
        }
    }
}

TEST(VolumeViewTest, ReconstructionTest) {
    const std::string equation = "euler3";
    size_t N = 5;
    auto deviceConfiguration = alsfvm::make_shared<alsfvm::DeviceConfiguration>("cpu");
    auto memoryFactory = alsfvm::make_shared<alsfvm::memory::MemoryFactory>(deviceConfiguration);
    auto volumeFactory = alsfvm::make_shared<alsfvm::volume::VolumeFactory>(equation, memoryFactory);

    auto volume = volumeFactory->createConservedVolume(N, 1, 1);

    for (size_t i = 0; i < N; ++i) {
        for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
            volume->getScalarMemoryArea(var)->getPointer()[i] = i*N + var;
        }
    }

    alsfvm::simulator::SimulatorParameters simulatorParameters("burgers", "cpu");
    alsfvm::grid::Grid grid({ 0.,0.,0. }, { 1.,0.,0. }, { N,1,1 });
    alsfvm::reconstruction::ReconstructionFactory reconstructionFactory;
    
    auto reconstruction = reconstructionFactory.createReconstruction("none", equation, simulatorParameters, memoryFactory,
        grid, deviceConfiguration);


    auto left = volumeFactory->createConservedVolume(N, 1, 1);
    auto right = volumeFactory->createConservedVolume(N, 1, 1);


    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        alsfvm::volume::Volume view(*volume, { var }, { "u" });
        alsfvm::volume::Volume viewLeft(*left, { var }, { "u" });

        alsfvm::volume::Volume viewRight(*right, { var }, { "u" });

        reconstruction->performReconstruction(view, 0, 0, viewLeft, viewRight);
    }

    for (size_t var = 0; var < volume->getNumberOfVariables(); ++var) {
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ(i*N + var, left->getScalarMemoryArea(var)->getPointer()[i]);
            ASSERT_EQ(i*N + var, right->getScalarMemoryArea(var)->getPointer()[i]);
        }
    }
}
