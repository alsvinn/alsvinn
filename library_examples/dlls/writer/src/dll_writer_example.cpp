#include "dll_writer_example.hpp"
#include "parameters.hpp"
#include "data.hpp"
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>

// Simple macro to print parameters
#define PRINT_PARAM(X) std::cout << "Value of " << #X << " is " << X << std::endl
extern "C" {

    DLL_WRITER_EXPORT void* create(const char* simulator_name,
        const char* simulator_version, void* parameters) {

        std::cout << "In create" << std::endl;

        PRINT_PARAM(simulator_name);
        PRINT_PARAM(simulator_version);
        PRINT_PARAM(parameters);

        return static_cast<void*>(new MyData);
    }

    DLL_WRITER_EXPORT void delete_data(void* data) {
        std::cout << "In delete_data" << std::endl;
        PRINT_PARAM(data);

        delete static_cast<MyData*>(data);
    }

    DLL_WRITER_EXPORT void write_data(void* data, void* parameters, double time,
        const char* variable_name, const double* variable_data, int nx, int ny, int nz,
        int ngx, int ngy, int ngz, double ax, double ay, double az, double bx,
        double by, double bz, int gpu_number ) {
        std::cout << "In write" << std::endl;

        PRINT_PARAM(data);
        PRINT_PARAM(parameters);
        PRINT_PARAM(time);
        PRINT_PARAM(variable_name);
        PRINT_PARAM(variable_data);

        PRINT_PARAM(nx);
        PRINT_PARAM(ny);
        PRINT_PARAM(nz);

        PRINT_PARAM(ngx);
        PRINT_PARAM(ngy);
        PRINT_PARAM(ngz);

        PRINT_PARAM(ax);
        PRINT_PARAM(ay);
        PRINT_PARAM(az);

        PRINT_PARAM(bx);
        PRINT_PARAM(by);
        PRINT_PARAM(bz);

        PRINT_PARAM(gpu_number);

        auto my_data = static_cast<MyData*>(data);
        auto my_parameters = static_cast<MyParameters*>(parameters);
        // In the write function, we will just write the data to a text file
        // readable by numpy
        auto output_name = my_parameters->getParameter("basename")
            + "_" + variable_name + "_"
            + std::to_string(my_data->getCurrentTimestep()) + ".txt";
        std::ofstream out_file(output_name);
        // Set highest possible precision, this way we are sure we are
        out_file << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

        // ignoring ghost cells (ngz is number of ghost cells in z direction)
        for (int z = ngz; z < nz + ngz; ++z) {
            // ignoring ghost cells (ngy is number of ghost cells in y direction)
            for (int y = ngy; y < ny + ngy; ++y) {
                // ignoring ghost cells (ngx is number of ghost cells in x direction)
                for (int x = ngx; x < nx + ngx; ++x) {
                    const auto index = z * (nx + 2 * ngx) * (ny + 2 * ngy) + y * (nx + 2 * ngx) + x;
                    out_file << variable_data[index] << "\n";
                }
            }

        }




    }

    DLL_WRITER_EXPORT void* make_parameters() {
        std::cout << "In make_parameters" << std::endl;

        return static_cast<void*>(new MyParameters());

    }

    DLL_WRITER_EXPORT void delete_parameters(void* parameters) {
        std::cout << "In delete_parameters" << std::endl;

        PRINT_PARAM(parameters);


        delete static_cast<MyParameters*>(parameters);
    }

    DLL_WRITER_EXPORT bool needs_data_on_host(void* data, void* parameters) {
        std::cout << "in needs_data_on_host" << std::endl;

        PRINT_PARAM(data);
        PRINT_PARAM(parameters);

        return true;

    }

    DLL_WRITER_EXPORT void set_parameter(void* parameters, const char* key,
        const char* value) {

        std::cout << "In set_parameter" << std::endl;

        auto my_parameters = static_cast<MyParameters*>(parameters);

        PRINT_PARAM(parameters);
        PRINT_PARAM(key);
        PRINT_PARAM(value);

        my_parameters->setParameter(key, value);
    }

    DLL_WRITER_EXPORT void set_mpi_comm(void* data, void* parameters,
        MPI_Comm communicator) {

        auto my_parameters = static_cast<MyParameters*>(parameters);
        std::cout << "In set_mpi_comm" << std::endl;

        PRINT_PARAM(parameters);
        PRINT_PARAM(data);
        PRINT_PARAM(communicator);

        my_parameters->setMPIComm(communicator);

    }

    DLL_WRITER_EXPORT void new_timestep(void* data, void* parameters, double time,
        int timestep_number) {
        std::cout << "in new_timestep" << std::endl;

        PRINT_PARAM(data);
        PRINT_PARAM(parameters);
        PRINT_PARAM(time);
        PRINT_PARAM(timestep_number);

        auto my_data = static_cast<MyData*>(data);

        my_data->setCurrentTimestep(timestep_number);

    }


    DLL_WRITER_EXPORT void end_timestep(void* data, void* parameters, double time,
        int timestep_number) {
        std::cout << "in end_timestep" << std::endl;

        PRINT_PARAM(data);
        PRINT_PARAM(parameters);
        PRINT_PARAM(time);
        PRINT_PARAM(timestep_number);

    }

}
