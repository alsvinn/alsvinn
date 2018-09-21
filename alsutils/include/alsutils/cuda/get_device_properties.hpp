/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#ifdef ALSVINN_HAVE_CUDA
    #include <cuda_runtime.h>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string/join.hpp>
#include "alsutils/cuda/cuda_safe_call.hpp"
#include "alsutils/log.hpp"
namespace alsutils {
namespace cuda {
//! Returns the information of cudaGetDeviceProperties for the current device
//! See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
//! for more information
inline boost::property_tree::ptree getDeviceProperties() {

    boost::property_tree::ptree propertiesTree;
    int currentDevice = -1;
#ifdef ALSVINN_HAVE_CUDA

    try {
        CUDA_SAFE_CALL_SILENT(cudaGetDevice(&currentDevice));


        cudaDeviceProp properties;
        CUDA_SAFE_CALL_SILENT(cudaGetDeviceProperties(&properties, currentDevice));

        propertiesTree.add("name", properties.name);
        propertiesTree.add("totalGlobalMem", properties.totalGlobalMem);
        propertiesTree.add("sharedMemPerBlock", properties.sharedMemPerBlock);
        propertiesTree.add("regsPerBlock", properties.regsPerBlock);
        propertiesTree.add("warpSize", properties.warpSize);
        propertiesTree.add("memPitch", properties.memPitch);
        propertiesTree.add("maxThreadsPerBlock", properties.maxThreadsPerBlock);
        propertiesTree.add("maxThreadsDim",
            boost::algorithm::join(
                std::vector<std::string>({std::to_string(properties.maxThreadsDim[0]),
                        std::to_string(properties.maxThreadsDim[1]),
                        std::to_string(properties.maxThreadsDim[2])}),
                ", "));
        propertiesTree.add("maxGridSize",
            boost::algorithm::join(
                std::vector<std::string>({std::to_string(properties.maxGridSize[0]),
                        std::to_string(properties.maxGridSize[1]),
                        std::to_string(properties.maxGridSize[2])}),
                ", "));
        propertiesTree.add("totalConstMem", properties.totalConstMem);
        propertiesTree.add("major", properties.major);
        propertiesTree.add("minor", properties.minor);
        propertiesTree.add("clockRate", properties.clockRate);
        propertiesTree.add("textureAlignment", properties.textureAlignment);
        propertiesTree.add("deviceOverlap", properties.deviceOverlap);
        propertiesTree.add("multiProcessorCount", properties.multiProcessorCount);
        propertiesTree.add("kernelExecTimeoutEnabled",
            properties.kernelExecTimeoutEnabled);
        propertiesTree.add("integrated", properties.integrated);
        propertiesTree.add("canMapHostMemory", properties.canMapHostMemory);
        propertiesTree.add("computeMode", properties.computeMode);
        propertiesTree.add("concurrentKernels", properties.concurrentKernels);
        propertiesTree.add("ECCEnabled", properties.ECCEnabled);
        propertiesTree.add("pciBusID", properties.pciBusID);
        propertiesTree.add("pciDeviceID", properties.pciDeviceID);
        propertiesTree.add("tccDriver", properties.tccDriver);




    } catch (std::runtime_error& e) {
        ALSVINN_LOG(WARNING,
            "(Ignore this if you are not running with CUDA). Failed getting CUDA GPU properties: "
            <<
            e.what());
    }

#endif

    return propertiesTree;
}
}
}
