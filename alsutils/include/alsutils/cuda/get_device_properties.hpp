#pragma once
#include <cuda_runtime.h>
#include <boost/property_tree/ptree.hpp>
namespace alsutils {
  namespace cuda {
    //! Returns the information of cudaGetDeviceProperties for the current device
    //! See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
    //! for more information
    inline boost::property_tree::ptree getDeviceProperties() {
      boost::property_tree::ptree propertiesTree;
      int currentDevice = -1;
      cudaGetDevice(&currentDevice);
      

      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, currentDevice);
      
      propertiesTree.add("name", properties.name);
      propertiesTree.add("totalGlobalMem", properties.totalGlobalMem);
      propertiesTree.add("regsPerBlock", properties.regsPerBlock);
      
      return propertiesTree;
    }
  }
}
