#include "alsfvm/io/DLLWriter.hpp"
#include <boost/algorithm/algorithm.hpp>
#include <boost/dll.hpp>
#include <boost/algorithm/string.hpp>
#include "alsutils/config.hpp"

namespace alsfvm {
namespace io {

DLLWriter::DLLWriter(const std::string& basename,
    const Parameters& parameters,
    alsutils::mpi::ConfigurationPtr mpiConfigration) {


    auto filename = parameters.getString("library");
    auto createFunctionName = parameters.getString("create_function");
    auto makeParametersName = parameters.getString("make_parameters_function");
    auto setMpiCommName = parameters.getString("set_mpi_comm_funtion");

    if (boost::algorithm::to_lower_copy(makeParametersName) != "none") {
        auto makeParametersFunction = boost::dll::import <void* ()>(filename,
                makeParametersName);
        parametersStruct = makeParametersFunction();

        auto setParameterFunctionName = parameters.getString("set_parameter_function");

        auto setParameterFunction = boost::dll::import
            <void(void*, const char*, const char*)>(filename,
                setParameterFunctionName);

        auto deleteParametersFunctionName =
            parameters.getString("delete_parameters_function");

        deleteParametersFunction = boost::dll::import
            <void(void*)>(filename,
                deleteParametersFunctionName);

        for (auto key : parameters.getKeys()) {

            setParameterFunction(parametersStruct, key.c_str(),
                parameters.getString(key).c_str());
        }
    }

    auto createFunction =
        boost::dll::import<void* (const char*, const char*, void*)>(filename,
            createFunctionName);

    dllData = createFunction("alsvinn",
            (std::string("https://github.com/alsvinn/alsvinn git") +
                alsutils::getVersionControlID()).c_str(),
            parametersStruct
        );

    auto newTimestepFunctionName =
        parameters.getString("new_timestep_function_name");

    if (boost::algorithm::to_lower_copy(newTimestepFunctionName) != "none") {
        newTimestepFunction = boost::dll::import<void(DLLData, DLLData, real, int)>
            (filename,
                newTimestepFunctionName);
    }

    auto endTimestepFunctionName =
        parameters.getString("end_timestep_function_name");


    if (boost::algorithm::to_lower_copy(endTimestepFunctionName) != "none") {
        endTimestepFunction = boost::dll::import<void(DLLData, DLLData, real, int)>
            (filename,
                endTimestepFunctionName);

    }

    auto setMpiCommFunctionName = parameters.getString("set_mpi_comm_function");

    if (boost::algorithm::to_lower_copy(setMpiCommFunctionName) != "none") {
        auto setMpiCommFunction = boost::dll::import<void(DLLData, DLLData, MPI_Comm)>
            (filename, setMpiCommFunctionName);

        setMpiCommFunction(dllData, parametersStruct,
            mpiConfigration->getCommunicator());
    }

    auto needsDataOnHostFunctionName =
        parameters.getString("needs_data_on_host_function");

    if (boost::algorithm::to_lower_copy(needsDataOnHostFunctionName) != "none") {
        auto needsDataOnHostFunction = boost::dll::import<bool(void*, void*)>(filename,
                needsDataOnHostFunctionName);

        needsDataOnHost = needsDataOnHostFunction(dllData, parametersStruct);
    }
}

void DLLWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables, const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    if (needsDataOnHost && !conservedVariables.getScalarMemoryArea(0)->isOnHost()) {

    }

    if (newTimestepFunction) {
        newTimestepFunction(dllData, parametersStruct,
            timestepInformation.getCurrentTime(),
            timestepInformation.getNumberOfStepsPerformed());
    }

    const real ax = grid.getOrigin().x;
    const real ay = grid.getOrigin().y;
    const real az = grid.getOrigin().z;

    const real bx = grid.getTop().x;
    const real by = grid.getTop().y;
    const real bz = grid.getTop().z;

    const int ngx = int(conservedVariables.getNumberOfXGhostCells());
    const int ngy = int(conservedVariables.getNumberOfYGhostCells());
    const int ngz = int(conservedVariables.getNumberOfZGhostCells());


    const int nx = int(conservedVariables.getNumberOfXCells());
    const int ny = int(conservedVariables.getNumberOfYCells());
    const int nz = int(conservedVariables.getNumberOfZCells());


    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        writeFunction(dllData, parametersStruct,
            timestepInformation.getCurrentTime(),
            conservedVariables.getName(var).c_str(),
            conservedVariables.getScalarMemoryArea(var)->getPointer(),
            nx, ny, nz,
            ngx, ngy, ngz,
            ax, ay, az,
            bx, by, bz,
            int(!conservedVariables.getScalarMemoryArea(var)->isOnHost()) - 1);
    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        writeFunction(dllData, parametersStruct,
            timestepInformation.getCurrentTime(),
            extraVariables.getName(var).c_str(),
            extraVariables.getScalarMemoryArea(var)->getPointer(),
            nx, ny, nz,
            ngx, ngy, ngz,
            ax, ay, az,
            bx, by, bz,
            int(!extraVariables.getScalarMemoryArea(var)->isOnHost()) - 1);
    }

    if (endTimestepFunction) {
        endTimestepFunction(dllData, parametersStruct,
            timestepInformation.getCurrentTime(),
            timestepInformation.getNumberOfStepsPerformed());
    }

    // remember, signature is
    //real write_function(void* data,
    //  void* parameters,
    //  real time,
    //  const char* variable_name,
    //  const real* variable_data,
    //  int nx, int ny, int nz,
    //  int ngx, int ngy, int ngz,
    //  real ax, real ay, real az,
    //  real bx, real by, real bz,
    //  int gpu_number );




}

void DLLWriter::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    if (deleteFunction) {
        deleteFunction(dllData);
    }

    if (deleteParametersFunction) {
        deleteParametersFunction(parametersStruct);
    }
}

}
}
