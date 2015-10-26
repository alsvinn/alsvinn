"""
Reads the data from hdf5 files using the vizschema layout
"""
import numpy
import h5py
class VizSchemaReader(object):
    def __init__(self, filename):
        self.filename = filename

    def read_dataset(self, datasetname):


        print("Reading from %s" % self.filename)
        with h5py.File(self.filename) as f:
            dataset = f[datasetname]
            data = numpy.zeros(dataset.shape)

            dataset.read_direct(data)

            while data.shape[-1] == 1:
                data.shape = data.shape[:-1]
            return data
