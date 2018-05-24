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

#include "alsfvm/mpi/CudaCartesianCellExchanger.hpp"
#include "alsutils/mpi/safe_call.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
namespace alsfvm {
namespace mpi {

namespace {

__global__ void extractSideDevice(memory::View<real> output,
    memory::View<const real> input,
    ivec3 start, ivec3 end) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = (index / nx) / ny;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int inputX = x + start.x;
    const int inputY = y + start.y;
    const int inputZ = z + start.z;

    output.at(x, y, z) = input.at(inputX, inputY, inputZ);

}


__global__ void insertSideDevice(memory::View<real> output,
    memory::View< real> input,
    ivec3 start, ivec3 end) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    const int nx = end.x - start.x;
    const int ny = end.y - start.y;
    const int nz = end.z - start.z;

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = (index / nx) / ny;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int outputX = x + start.x;
    const int outputY = y + start.y;
    const int outputZ = z + start.z;

    output.at(outputX, outputY, outputZ) = input.at(x, y, z);

}

}
CudaCartesianCellExchanger::CudaCartesianCellExchanger(ConfigurationPtr&
    configuration, const ivec6& neighbours)
    : configuration(configuration), neighbours(neighbours) {


}

RequestContainer CudaCartesianCellExchanger::exchangeCells(
    volume::Volume& outputVolume,
    const volume::Volume& inputVolume) {

    const int dimensions = inputVolume.getDimensions();

    if (buffers.size() == 0) {
        makeBuffers(inputVolume);
        makeStreams(inputVolume);

        receiveRequests.resize(inputVolume.getNumberOfVariables());
        sendRequests.resize(inputVolume.getNumberOfVariables());

        for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
            receiveRequests[var].resize(2 * dimensions);
            sendRequests[var].resize(2 * dimensions);
        }
    }


    extractSides(inputVolume);

    auto oppositeSide = [&](int side) {
        const int i = side % 2;
        return (i + 1) % 2 + (side / 2) * 2;
    };

    RequestContainer container;


    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {

        for (int side = 0; side < 2 * dimensions; ++side) {
            if (hasSide(side)) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(memoryStreams[var][side]));
                sendRequests[var][side] = (Request::isend(cpuBuffersSend[var][side],
                            cpuBuffersSend[var][side].size(),
                            MPI_DOUBLE, neighbours[side],
                            var * 6 + side,
                            *configuration));
            }

            if (hasSide(oppositeSide(side))) {
                receiveRequests[var][oppositeSide(side)] = Request::ireceive(
                        cpuBuffersReceive[var][oppositeSide(side)],
                        cpuBuffersReceive[var][oppositeSide(side)].size(),
                        MPI_DOUBLE, neighbours[oppositeSide(side)],
                        var * 6 + side,
                        *configuration);
            }
        }
    }


    insertSides(outputVolume);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    RequestContainer emptyContainer;
    return emptyContainer;
}

int CudaCartesianCellExchanger::getNumberOfActiveSides() const {
    int activeSides = 0;

    for (int i = 0; i < 6; ++i) {
        if (hasSide(i)) {
            activeSides++;
        }
    }

    return activeSides;
}

real CudaCartesianCellExchanger::max(real number) {
    real max;
    MPI_SAFE_CALL(MPI_Allreduce(&number, &max, 1, MPI_DOUBLE, MPI_MAX,
            configuration->getCommunicator()));
    return max;
}
bool CudaCartesianCellExchanger::hasSide(int side) const {
    return neighbours[side] != -1;
}

void CudaCartesianCellExchanger::extractSide(const ivec3& start,
    const ivec3& end,
    int side,
    const volume::Volume& inputVolume) {
    for (int var  = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        const auto diff = end - start;
        const int size = diff.x * diff.y * diff.z;

        if (size <= 0) {
            return;
        }

        const int numberOfThreads = 512;

        extractSideDevice <<< (size + numberOfThreads - 1) / numberOfThreads,
                          numberOfThreads,
                          0, memoryStreams[var][side] >>> (buffers[var][side]->getView(),
                              inputVolume.getScalarMemoryArea(var)->getView(),
                              start,
                              end);


        CUDA_SAFE_CALL(cudaMemcpyAsync(cpuBuffersSend[var][side].data(),
                buffers[var][side]->getPointer(),
                buffers[var][side]->getSize()*sizeof(real),
                cudaMemcpyDeviceToHost,
                memoryStreams[var][side]));


    }


}

ivec6 CudaCartesianCellExchanger::getNeighbours() const {
    return neighbours;
}
void CudaCartesianCellExchanger::extractSides(const volume::Volume&
    inputVolume) {
    const int nx = inputVolume.getTotalNumberOfXCells();
    const int ny = inputVolume.getTotalNumberOfYCells();
    const int nz = inputVolume.getTotalNumberOfZCells();

    const int ngx = inputVolume.getNumberOfXGhostCells();
    const int ngy = inputVolume.getNumberOfYGhostCells();
    const int ngz = inputVolume.getNumberOfZGhostCells();

    const int dimensions = inputVolume.getDimensions();

    if (hasSide(0)) {
        extractSide({ngx, 0, 0}, {2 * ngx, ny, nz}, 0, inputVolume);
    }

    if (hasSide(1)) {

        extractSide({nx - 2 * ngx, 0, 0}, {nx - ngx, ny, nz}, 1, inputVolume);
    }

    if (dimensions > 1) {

        if (hasSide(2)) {

            extractSide({0, ngy, 0}, {nx, 2 * ngy, nz}, 2, inputVolume);
        }

        if (hasSide(3)) {

            extractSide({0, ny - 2 * ngy, 0}, {nx, ny - ngy, nz}, 3, inputVolume);
        }

        if (dimensions > 2) {
            if (hasSide(4)) {

                extractSide({0, 0, ngz}, {nx, ny, 2 * ngz}, 4, inputVolume);
            }

            if (hasSide(5)) {

                extractSide({0, 0, nz - 2 * ngz}, {nx, ny, nz - ngz}, 5, inputVolume);
            }
        }

    }

}


void CudaCartesianCellExchanger::insertSide(const ivec3& start,
    const ivec3& end,
    int side,
    volume::Volume& outputVolume) {
    for (int var  = 0; var < outputVolume.getNumberOfVariables(); ++var) {


        const auto diff = end - start;
        const int size = diff.x * diff.y * diff.z;

        if (size == 0) {
            return;
        }


        //sendRequests[var][side]->wait();
        receiveRequests[var][side]->wait();
        CUDA_SAFE_CALL(cudaMemcpyAsync(buffers[var][side]->getPointer(),
                cpuBuffersReceive[var][side].data(),
                buffers[var][side]->getSize()*sizeof(real),
                cudaMemcpyHostToDevice,
                memoryStreams[var][side]));

        const int numberOfThreads = 512;
        insertSideDevice <<< (size + numberOfThreads - 1) / numberOfThreads,
                         numberOfThreads,
                         0, memoryStreams[var][side] >>> (
                             outputVolume.getScalarMemoryArea(var)->getView(),
                             buffers[var][side]->getView(),
                             start,
                             end);

    }
}


void CudaCartesianCellExchanger::insertSides( volume::Volume& outputVolume) {
    const int nx = outputVolume.getTotalNumberOfXCells();
    const int ny = outputVolume.getTotalNumberOfYCells();
    const int nz = outputVolume.getTotalNumberOfZCells();

    const int ngx = outputVolume.getNumberOfXGhostCells();
    const int ngy = outputVolume.getNumberOfYGhostCells();
    const int ngz = outputVolume.getNumberOfZGhostCells();


    const int dimensions = outputVolume.getDimensions();

    if (hasSide(0)) {
        insertSide({0, 0, 0}, {ngx, ny, nz}, 0, outputVolume);
    }

    if (hasSide(1)) {
        insertSide({nx - ngx, 0, 0}, {nx, ny, nz}, 1, outputVolume);
    }

    if (dimensions > 1) {

        if (hasSide(2)) {
            insertSide({0, 0, 0}, {nx, ngy, nz}, 2, outputVolume);
        }

        if (hasSide(3)) {
            insertSide({0, ny - ngy, 0}, {nx, ny, nz}, 3, outputVolume);
        }

        if (dimensions > 2 ) {
            if (hasSide(4)) {
                insertSide({0, 0, 0}, {nx, ny, ngz}, 4, outputVolume);
            }

            if (hasSide(5)) {
                insertSide({0, 0, nz - ngz}, {nx, ny, nz}, 5, outputVolume);
            }
        }
    }

}


void CudaCartesianCellExchanger::makeStreams(const volume::Volume&
    inputVolume) {
    memoryStreams.resize(inputVolume.getNumberOfVariables());

    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        memoryStreams[var].resize(6);
        int dimensions = inputVolume.getDimensions();

        for (int side = 0; side < 2 * dimensions; ++side) {
            CUDA_SAFE_CALL(cudaStreamCreate(&memoryStreams[var][side]));
            //memoryStreams[var][side]=0;
        }
    }
}

void CudaCartesianCellExchanger::makeBuffers(const volume::Volume&
    inputVolume) {
    buffers.resize(inputVolume.getNumberOfVariables());
    cpuBuffersSend.resize(buffers.size());
    cpuBuffersReceive.resize(buffers.size());

    for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
        buffers[var].resize(6);
        cpuBuffersSend[var].resize(6);
        cpuBuffersReceive[var].resize(6);

        for (int side = 0; side < 6; ++side) {
            if (hasSide(side)) {
                const int nx = (side > 1) * inputVolume.getTotalNumberOfXCells() +
                    (side < 2) * inputVolume.getNumberOfXGhostCells();

                const int ny = (side != 2) * (side != 3) * inputVolume.getTotalNumberOfYCells()
                    +
                    ((side == 2) + (side == 3)) * inputVolume.getNumberOfYGhostCells();


                const int nz = (side != 4) * (side != 5) * inputVolume.getTotalNumberOfZCells()
                    +
                    ((side == 4) + (side == 5)) * inputVolume.getNumberOfZGhostCells();



                buffers[var][side] = alsfvm::make_shared<alsfvm::cuda::CudaMemory<real>>(nx, ny,
                        nz);

                cpuBuffersSend[var][side].resize(nx * ny * nz, 0);
                cpuBuffersReceive[var][side].resize(nx * ny * nz, 0);
                //alsfvm::make_shared<alsfvm::memory::HostMemory<real>>(nx, ny, nz);
            }
        }
    }
}
}
}
