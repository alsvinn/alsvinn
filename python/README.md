The Alsvinn Python Interface
------------------------------
This is meant as a simple interface for running and plotting alsvinn related
data. This is *not* a python wrapper for the alsvinn libraries, but rather
a python API that lets you run the alsvinncli program (and alsuqcli).

The alsvinn/config.py.in is meant to be build through the cmake system (it will include all the paths). Include the folder ```<build folder>/python``` in your ```PYTHONPATH```  and then import ```alsvinn```.

## Example program

    # from the build directory
    PYTHONPATH=$PYTHONPATH:python ipython
    
    In [1]: import alsvinn
    In [2]: alsvinnObject = alsvinn.run()
    In [3]: alsvinnObject.plot('rho', 1)
    
    