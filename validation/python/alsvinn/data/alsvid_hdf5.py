"""
Loads and stores the data from alsvid (hdf5 version)
"""
import numpy
import h5py

from vizschema_reader import VizSchemaReader


class AlsvidReader(object):
    def __init__(self, filename):
        self.filename = filename

    def __to_alsvid(self, datasetname_alsvinn):
        return datasetname_alsvinn.replace("x","0").replace("y", "1").replace("z", "2")

    def read_dataset(self, datasetname):
        datasetname_alsvid = self.__to_alsvid(datasetname)
        return VizSchemaReader(self.filename).read_dataset(datasetname_alsvid)
