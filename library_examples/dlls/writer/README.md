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

The DLL is rather simple and just writes the output to text file (readable by ```numpy.loadtxt``` in Python/numpy)

## Some comments on the code

Alsvinn always uses DLLs that have a pure C interface (so no C++ types are allowed in the function signature). However, we usually want to use proper C++ types in the DLL. The way to do this is by casting, so eg. when we make the parameter struct, we just do

```cpp
void *make_parameters() {
     return static_cast<void*>(new MyParameters());
}
```
then when we use it, we cast it back to ```MyParameters```, in eg. ```set_parameter```

```cpp
auto my_parameters = static_cast<MyParameters*>(parameters);
// here we can use any function of MyParameters
```