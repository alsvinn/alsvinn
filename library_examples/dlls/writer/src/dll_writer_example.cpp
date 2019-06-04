#include "dll_writer_example.hpp"
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

        return nullptr;
    }

    DLL_WRITER_EXPORT void delete_data(void* data) {
        std::cout << "In delete_data" << std::endl;
        PRINT_PARAM(data);

    }

    DLL_WRITER_EXPORT void write(void* data, void* parameters, double time,
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




    }

    DLL_WRITER_EXPORT void* make_parameters() {
        std::cout << "In make_parameters" << std::endl;

        return nullptr;

    }

    DLL_WRITER_EXPORT void delete_parameters(void* parameters) {
        std::cout << "In delete_parameters" << std::endl;

        PRINT_PARAM(parameters);


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

        PRINT_PARAM(parameters);
        PRINT_PARAM(key);
        PRINT_PARAM(value);
    }

    DLL_WRITER_EXPORT void set_mpi_comm(void* parameters, void* data,
        MPI_Comm communicator) {

        std::cout << "In set_mpi_comm" << std::endl;

        PRINT_PARAM(parameters);
        PRINT_PARAM(data);
        PRINT_PARAM(communicator);

    }

    DLL_WRITER_EXPORT void new_timestep(void* data, void* parameters, double time,
        int timestep_number) {
        std::cout << "in new_timestep" << std::endl;

        PRINT_PARAM(data);
        PRINT_PARAM(parameters);
        PRINT_PARAM(time);
        PRINT_PARAM(timestep_number);

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
