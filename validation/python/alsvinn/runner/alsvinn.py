import xml.etree.ElementTree
import os
import configuration

class Alsvinn(object):
    def __init__(self, basefile, configuration):
        self.basefile = basefile
        self.configuration = configuration


    def __call__(self, resolutions):
        basenames = []
        for resolution in resolutions:
            (xml_file, basename) = self.__make_xml(resolution)

            self.__run_alsvinn(xml_file)

            basenames.append((basename, "alsvinn"))
        return basenames



    def __make_xml(self, resolution):
        document = xml.etree.ElementTree.ElementTree()

        document.parse(self.basefile)
        root = document.getroot()

        resolution_str = " ".join(map(str, resolution))
        resolution_file_str = "_".join(map(str, resolution))
        xml_file = self.basefile[0:-4] + resolution_file_str + ".xml"
        print("xml_file = %s" % xml_file)

        root.findall('./grid/dimension')[0].text = resolution_str

        for key, value in self.configuration.iteritems():
            print("setting %s = %s" % (key, value))
            root.findall("./%s" % key)[0].text =  value

        basename = root.findall('./writer/basename')[0]
        basename.text = basename.text + resolution_file_str

        timesteps = int(root.findall("./writer/numberOfSaves")[0].text)


        document.write(xml_file)

        return (xml_file, "%s_%d.h5" % (basename.text, timesteps))

    def __run_alsvinn(self, xml_file):
        command_to_run = "alsvinncli %s" % xml_file
        print(command_to_run)
        os.system(command_to_run)


def run_configurations(basefile, configurations, resolutions, consume_function):
    for config in configuration.make_configurations(configurations):
        runner = Alsvinn(basefile, config)
        output_files = runner(resolutions)
        consume_function(config, output_files)
