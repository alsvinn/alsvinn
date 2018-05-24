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

#include <gtest/gtest.h>
#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/io/netcdf_utils.hpp"
#include "alsuq/io/MPIWriterFactory.hpp"
#include <netcdf.h>
#include "alsutils/log.hpp"
using namespace alsfvm;

TEST(NetCDFMPITest, TestSimpleVolume) {
    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>("cpu");

    auto memoryFactory = make_shared<memory::MemoryFactory>(deviceConfiguration);

    volume::VolumeFactory volumeFactory("euler2", memoryFactory);
    size_t nx = 3, ny = 4, nz = 5, ng = 2;

    auto conservedVolume = volumeFactory.createConservedVolume(nx, ny, nz, ng);
    auto extraVolume = volumeFactory.createExtraVolume(nx, ny, nz, ng);
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    ALSVINN_LOG(INFO, "Creating input data");

    for (size_t var = 0; var < conservedVolume->getNumberOfVariables(); ++var) {
        auto memory = conservedVolume->getScalarMemoryArea(var);
        auto view = memory->getView();

        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++ x) {
                    view.at(x + 2, y + 2, z + 2) = var * nx * ny * nz + z * ny * nx + y * nx + x;
                }
            }

        }
    }


    for (size_t var = 0; var < extraVolume->getNumberOfVariables(); ++var) {
        auto memory = extraVolume->getScalarMemoryArea(var);

        auto view = memory->getView();

        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++ x) {
                    view.at(x + 2, y + 2, z + 2) = (var + 4) * nx * ny * nz + z * ny * nx + y * nx +
                        x;
                }
            }

        }
    }

    ALSVINN_LOG(INFO, "Done writing input data");
    io::WriterFactory writerFactory;
    const std::string basename = "netcdf_test";
    auto writer = writerFactory.createWriter("netcdf", "netcdf_test");

    simulator::TimestepInformation timestepInformation;
    std::cout << "About to write" << std::endl;
    ALSVINN_LOG(INFO, "Writing file")
    writer->write(*conservedVolume, *extraVolume, grid, timestepInformation);
    ALSVINN_LOG(INFO, "Done writing");
    // Now we read it back in
    std::cout << "done writing" << std::endl;
    netcdf_raw_ptr file;
    NETCDF_SAFE_CALl(nc_open((basename + "_0.nc").c_str(), NC_NOWRITE, &file));



    ALSVINN_LOG(INFO, "Opened file");

    for (size_t var = 0; var < conservedVolume->getNumberOfVariables(); ++var) {
        netcdf_raw_ptr varId;
        NETCDF_SAFE_CALl(nc_inq_varid(file, conservedVolume->getName(var).c_str(),
                &varId));
        std::vector<double> data(nx * ny * nz, 0);

        NETCDF_SAFE_CALl(nc_get_var_double(file, varId, data.data()));

        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++ x) {
                    ASSERT_EQ(var * nx * ny * nz + z * ny * nx + y * nx + x,
                        data[z * nx * ny + y * nx + x]);
                }
            }

        }

    }

    for (size_t var = 0; var < extraVolume->getNumberOfVariables(); ++var) {
        netcdf_raw_ptr varId;
        NETCDF_SAFE_CALl(nc_inq_varid(file, extraVolume->getName(var).c_str(), &varId));
        std::vector<double> data(nx * ny * nz, 0);

        NETCDF_SAFE_CALl(nc_get_var_double(file, varId, data.data()));


        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++ x) {
                    ASSERT_EQ((var + 4)*nx * ny * nz + z * ny * nx + y * nx + x,
                        data[z * nx * ny + y * nx + x]);
                }
            }

        }
    }


    NETCDF_SAFE_CALl(nc_close(file));

}
