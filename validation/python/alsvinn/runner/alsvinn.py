import xml.etree.ElementTree
import os

class Alsvinn(object):
    def __init__(self, basefile, configuration):
        self.basefile = basefile
        self.configuration = configuration


    def __call__(self, resolutions):
        basenames = []
        for resolution in resolutions:
            (xml_file, basename) = self.__make_xml(resolution)

            self.__run_alsvinn(xml_file)

            basenames.append(basename)
        return basenames



    def __make_xml(self, resolution):
        document = xml.etree.ElementTree.parse(self.basefile)
        root = document.getroot()

        resolution_str = " ".join(map(str, resolution))
        root.findall('./fvm/grid/resolution')[0].set(resolution_str)

        for key, value in self.configuration.iteritems():
            root.findall("./fvm/%s" % key)[0].set(value)

        basename = root.findall('./fvm/writer/basename')[0]
        basename.set(basename.text + resolution_str)

        xml_file = self.basefile[0:-4] + resolution_str + ".xml"

        document.write(xml_file)

        return (xml_file, basename.text)

    def __run_alsvinn(self, xml_file):
        os.system("alsvinncli %s" % xml_file)
