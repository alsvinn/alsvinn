"""
Loads and stores the data from alsvinn (hdf5 version)
"""
import numpy
import h5py

from vizschema_reader import VizSchemaReader


class AlsvinnReader(object):
    def __init__(self, filename):
        self.filename = filename


    def read_dataset(self, datasetname):
        return VizSchemaReader(self.filename).read_dataset(datasetname)

