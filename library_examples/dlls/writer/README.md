# Creating a DLL for writing data

In this example, we create a dll/shared library that be loaded dynamically from alsvinn on runtime.

To compile, run

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE
    make

To run

    # from the build folder where dll_writer_example.so existsd
    <path to alsuqcli>/alsuqcli ../examples/1d/sodshocktube.xml

