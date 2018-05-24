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
#include "alsfvm/mpi/CellExchanger.hpp"
#include "alsfvm/cuda/CudaMemory.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#include <thrust/host_vector.h>

namespace alsfvm {
namespace mpi {

//! Does the cell exchange for cuda. This class is supposed to be a placeholder
//! for when gpu direct is not available.
class CudaCartesianCellExchanger : public CellExchanger {
public:
    //! Constructs a new instance
    //!
    //! @param configuration a pointer to the current MPI configuration
    //! @param neighbours the list of processor neighbours for each side. Has
    //!                   the following format
    //!
    //! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
    //! -------|------------------|-----------------|-----------------
    //!    0   |       left       |     left        |    left
    //!    1   |       right      |     right       |    right
    //!    2   |     < not used > |     bottom      |    bottom
    //!    3   |     < not used > |     top         |    top
    //!    4   |     < not used > |   < not used >  |    front
    //!    5   |     < not used > |   < not used >  |    back
    CudaCartesianCellExchanger(ConfigurationPtr& configuration,
        const ivec6& neighbours);

    RequestContainer exchangeCells(volume::Volume& outputVolume,
        const volume::Volume& inputVolume) override;

    real max(real number) override;

    bool hasSide(int side) const;

    int getNumberOfActiveSides() const;

    ivec6 getNeighbours()  const override;
private:
    ConfigurationPtr configuration;
    ivec6 neighbours;

    // for each variable, for each side
    std::vector<std::vector<alsfvm::shared_ptr<cuda::CudaMemory<real> > > > buffers;

    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersSend;
    std::vector<std::vector<thrust::host_vector<real> > > cpuBuffersReceive;
    void makeBuffers(const volume::Volume& inputVolume);
    void makeStreams(const volume::Volume& inputVolume);

    void extractSides(const volume::Volume& inputVolume);
    void extractSide(const ivec3& start, const ivec3& end,
        int side,
        const volume::Volume& inputvolume);

    void insertSides(volume::Volume& outputVolume);
    void insertSide(const ivec3& start, const ivec3& end,
        int side,
        volume::Volume& outputVolume);


    std::vector<std::vector<cudaStream_t> > memoryStreams;

    std::vector<std::vector<RequestPtr> > receiveRequests;
    std::vector<std::vector<RequestPtr> > sendRequests;


};
} // namespace mpi
} // namespace alsfvm
