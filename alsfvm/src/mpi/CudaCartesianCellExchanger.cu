#include "alsfvm/mpi/CudaCartesianCellExchanger.hpp"
#include "alsutils/mpi/safe_call.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
namespace alsfvm { namespace mpi { 

    namespace {

        __global__ void extractSideDevice(memory::View<real> output,
                                    memory::View<const real> input,
                                    ivec3 start, ivec3 end) {
            const int index = threadIdx.x + blockIdx.x*blockDim.x;

            const int nx = end.x - start.x;
            const int ny = end.y - start.y;
            const int nz = end.z - start.z;

            const int x = index%nx;
            const int y = (index/nx)%ny;
            const int z = (index/nx)/ny;

            if (x >= nx || y>= ny || z >= nz) {
                return;
            }

            const int inputX = x + start.x;
            const int inputY = y + start.y;
            const int inputZ = z + start.z;

            output.at(x,y,z) = input.at(inputX, inputY, inputZ);

        }


        __global__ void insertSideDevice(memory::View<real> output,
                                    memory::View< real> input,
                                    ivec3 start, ivec3 end) {
            const int index = threadIdx.x + blockIdx.x*blockDim.x;

            const int nx = end.x - start.x;
            const int ny = end.y - start.y;
            const int nz = end.z - start.z;

            const int x = index%nx;
            const int y = (index/nx)%ny;
            const int z = (index/nx)/ny;

            if (x >= nx || y>= ny || z >= nz) {
                return;
            }

            const int outputX = x + start.x;
            const int outputY = y + start.y;
            const int outputZ = z + start.z;

            output.at(outputX,outputY, outputZ) = input.at(x, y, z);

        }

    }
    CudaCartesianCellExchanger::CudaCartesianCellExchanger(ConfigurationPtr &configuration, const ivec6 &neighbours)
        : configuration(configuration), neighbours(neighbours) {

    }

    RequestContainer CudaCartesianCellExchanger::exchangeCells(volume::Volume &outputVolume,
                                                               const volume::Volume &inputVolume) {
							       

        if (buffers.size() == 0){
            makeBuffers(inputVolume);
        }

        extractSides(inputVolume);

        auto oppositeSide = [&](int side) {
            const int i = side%2;
            return (i+1)%2 + (side/2)*2;
        };

        RequestContainer container;
        for(int side = 0; side < 6; ++side) {
            for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
                if (hasSide(side)) {
                    container.addRequest(Request::isend(*buffers[var][side], buffers[var][side]->getSize(),
                                                        MPI_DOUBLE, neighbours[side],
                                                        var*6+side,
                                                        *configuration));
                }

                if(hasSide(oppositeSide(side))) {
                    container.addRequest(Request::ireceive(*buffers[var][oppositeSide(side)], buffers[var][oppositeSide(side)]->getSize(),
                                                        MPI_DOUBLE, neighbours[oppositeSide(side)],
                                                        var*6+side,
                                                        *configuration));
                }
            }
        }


        container.waitForAll();

	insertSides(outputVolume);

	return container;
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
        MPI_SAFE_CALL(MPI_Allreduce(&number, &max, 1, MPI_DOUBLE, MPI_MAX, configuration->getCommunicator()));
        return max;
    }
    bool CudaCartesianCellExchanger::hasSide(int side) const {
        return neighbours[side]!=-1;
    }

    void CudaCartesianCellExchanger::extractSide(const ivec3 &start,
                                                 const ivec3 &end,
                                                 int side,
                                                 const volume::Volume &inputVolume) {
        for (int var  = 0; var < inputVolume.getNumberOfVariables(); ++var) {
            const auto diff = end - start;
            const int size = diff.x*diff.y*diff.z;
	    if (size == 0) {
	       return;
	    }	       

            const int numberOfThreads = 512;

            extractSideDevice<<<(size+numberOfThreads-1)/numberOfThreads, numberOfThreads>>>(buffers[var][side]->getView(),
                                                                                             inputVolume.getScalarMemoryArea(var)->getView(),
                                                                                             start,
            end);

        }
    }


    void CudaCartesianCellExchanger::extractSides(const volume::Volume &inputVolume) {
        const int nx = inputVolume.getTotalNumberOfXCells();
        const int ny = inputVolume.getTotalNumberOfYCells();
        const int nz = inputVolume.getTotalNumberOfZCells();

        const int ngx = inputVolume.getNumberOfXGhostCells();
        const int ngy = inputVolume.getNumberOfYGhostCells();
        const int ngz = inputVolume.getNumberOfZGhostCells();


        if (hasSide(0)) {

            extractSide({0,0,0},{ngx, ny, nz}, 0, inputVolume);
        }
        if (hasSide(1)) {

            extractSide({nx-ngx,0,0},{nx, ny, nz}, 1, inputVolume);
        }

        if (hasSide(2)) {

            extractSide({0,0,0},{nx, ngy, nz}, 2, inputVolume);
        }
        if (hasSide(3)) {

            extractSide({0,ny-ngy,0},{nx, ny, nz}, 3, inputVolume);
        }

        if(hasSide(4)) {

            extractSide({0,0,0},{nx, ny, ngz}, 4, inputVolume);
        }

        if(hasSide(5)) {

            extractSide({0,0,nz-ngz},{nx, ny, nz}, 5, inputVolume);
        }

    }


    void CudaCartesianCellExchanger::insertSide(const ivec3 &start,
                                                 const ivec3 &end,
                                                 int side,
                                                  volume::Volume &outputVolume) {
        for (int var  = 0; var < outputVolume.getNumberOfVariables(); ++var) {
            const auto diff = end - start;
            const int size = diff.x*diff.y*diff.z;
	    if (size == 0) {
	       return;
            }
            const int numberOfThreads = 512;
            insertSideDevice<<<(size+numberOfThreads-1)/numberOfThreads, numberOfThreads>>>(
                                                                                             outputVolume.getScalarMemoryArea(var)->getView(),
                                                                                              buffers[var][side]->getView(),
                                                                                             start,
            end);
        }
    }


    void CudaCartesianCellExchanger::insertSides( volume::Volume &outputVolume) {
        const int nx = outputVolume.getTotalNumberOfXCells();
        const int ny = outputVolume.getTotalNumberOfYCells();
        const int nz = outputVolume.getTotalNumberOfZCells();

        const int ngx = outputVolume.getNumberOfXGhostCells();
        const int ngy = outputVolume.getNumberOfYGhostCells();
        const int ngz = outputVolume.getNumberOfZGhostCells();


        if (hasSide(0)) {
            insertSide({0,0,0},{ngx, ny, nz}, 0, outputVolume);
        }
        if (hasSide(1)) {
            insertSide({nx-ngx,0,0},{nx, ny, nz}, 1, outputVolume);
        }

        if (hasSide(2)) {
            insertSide({0,0,0},{nx, ngy, nz}, 2, outputVolume);
        }
        if (hasSide(3)) {
            insertSide({0,ny-ngy,0},{nx, ny, nz}, 3, outputVolume);
        }

        if(hasSide(4)) {
            insertSide({0,0,0},{nx, ny, ngz}, 4, outputVolume);
        }

        if(hasSide(5)) {
            insertSide({0,0,nz-ngz},{nx, ny, nz}, 5, outputVolume);
        }

    }

    void CudaCartesianCellExchanger::makeBuffers(const volume::Volume &inputVolume) {
        buffers.resize(inputVolume.getNumberOfVariables());
        for (int var = 0; var < inputVolume.getNumberOfVariables(); ++var) {
            buffers[var].resize(6);
            for(int side = 0; side < 6; ++side) {
                if (hasSide(side)) {
                    const int nx = (side>1)*inputVolume.getTotalNumberOfXCells() +
                            (side<2)*inputVolume.getNumberOfXGhostCells();

                    const int ny = (side!=2)*(side!=3)*inputVolume.getTotalNumberOfYCells() +
                            ((side==2) + (side==3))*inputVolume.getNumberOfYGhostCells();


                    const int nz = (side!=4)*(side!=5)*inputVolume.getTotalNumberOfZCells() +
                            ((side==4) + (side==5))*inputVolume.getNumberOfZGhostCells();



                    buffers[var][side] = alsfvm::make_shared<alsfvm::cuda::CudaMemory<real>>(nx, ny, nz);
                }
            }
        }
    }
}
}
