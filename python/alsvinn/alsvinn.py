import alsvinn.config
import xml.dom.minidom
import numpy
import os
import dicttoxml
import copy
import shutil
import netCDF4
import matplotlib
import matplotlib.pyplot
class Alsvinn(object):
    def __init__(self, xml_file=None, configuration=None, alsvinncli=alsvinn.config.ALSVINNCLI_PATH,
                 prepend_alsvinncli=''):
        self.settings = {"fvm" : {}}
        self.fvmSettings = self.settings["fvm"]
        if xml_file != None:
            self.read_xml(xml_file)
        elif configuration != None:
            self.fvmSettings = configuration
        self.alsvinncli = alsvinncli
        self.prepend_alsvinncli = prepend_alsvinncli




    def __xml_item_func(self, parent_tag):
        return parent_tag[:-1]
    def set_fvm_value(self, key, value):
        self.fvmSettings[key] = value

    def set_initial_data(self, python_file, parameters):
        self.fvmSettings["initialData"] = {"python" : python_file}
        if parameters != None:
            self.fvmSettings["initialData"]["parameters"] = []
            for parameter_name in parameters.keys():
                parameter = {"name" : parameter_name, "length" : len(parameters[parameter_name])}
                parameter["values"] = []
                for value in parameters[key]:
                    parameter["value"].append(value)

                self.fvmSettings["initialData"]["parameters"].append(parameter)


    def remove_fvm_writer(self):
        self.fvmSettings["writer"] = None

    def set_equation_parameters(self, parameters):
        self.fvmSettings["equationParameters"] = parameters

    def set_fvm_writer(self, type, basename, number_of_saves):
        self.fvmSettings["writer"] = {}
        self.fvmSettings["writer"]["type"] = type
        self.fvmSettings["writer"]["basename"] = basename
        self.fvmSettings["writer"]["numberOfSaves"] = number_of_saves

    def set_cartesian_grid(self, lower_corner, upper_corner, dimensions):
        self.fvmSettings["grid"] = {"lowerCorner" : lower_corner,
                                    "upperCorner" : upper_corner,
                                    "dimension" : dimensions}


    def read_xml(self, xml_file):
        self.source_xml_path=xml_file
        document = xml.dom.minidom.parse(xml_file)
        configDocument = document.getElementsByTagName("config")[0]
        fvmDocument = configDocument.getElementsByTagName("fvm")[0]

        self.__readValues(self.fvmSettings, fvmDocument)

        # now we need to do some manual adjustments
        self.fvmSettings["grid"]["lowerCorner"] = self.__make_float(self.fvmSettings["grid"]["lowerCorner"])
        self.fvmSettings["grid"]["upperCorner"] = self.__make_float(self.fvmSettings["grid"]["upperCorner"])

        self.fvmSettings["grid"]["dimension"] = self.__make_int(self.fvmSettings["grid"]["dimension"])
        self.fvmSettings["endTime"] = self.__make_float(self.fvmSettings["endTime"])



    def set_lower_corner(self, corner):
        self.fvmSettings['grid']['lowerCorner'] = corner

    def set_upper_corner(self, corner):
        self.fvmSettings['grid']['upperCorner'] = corner

    def set_dimension(self, dim):
        self.fvmSettings['grid']['dimension'] = dim


    def __make_float(self, text):

        text = text.strip()
        splitted = text.split(" ")
        if len(splitted) == 1:
            return float(splitted[0])
        else:
            vector = numpy.zeros(len(splitted))
            for n in range(len(vector)):
                vector[n] = float(splitted[n])
        return vector

    def __make_int(self, text):
        text = text.strip()
        splitted = text.split(" ")
        if len(splitted) == 1:
            return int(splitted[0])
        else:
            vector = [0 for i in range(len(splitted))]
            for n in range(len(vector)):
                vector[n] = int(splitted[n])
        return vector

    def run(self, mpi_threads=1, multix=1,multiy=1,multiz=1):
        xmlFilename = self.fvmSettings["name"] + ".xml"

        settings = copy.deepcopy(self.settings)
        self.__make_string(settings["fvm"]["grid"], ["lowerCorner", "upperCorner", "dimension"])

        newPythonFile = self.get_name() + '.py'

        if settings["fvm"]["initialData"]["python"] != newPythonFile:
            shutil.copyfile(os.path.join(os.path.dirname(self.source_xml_path), settings["fvm"]["initialData"]["python"]), newPythonFile)
            settings["fvm"]["initialData"]["python"] = newPythonFile

        xmlObject = dicttoxml.dicttoxml(settings, custom_root='config', attr_type=False, item_func=self.__xml_item_func)
        #print (xmlObject)
        xmlDocument = xml.dom.minidom.parseString(xmlObject)
        with open (xmlFilename, "w") as f:
            f.write(xmlDocument.toprettyxml())

        if mpi_threads == 1:
            os.system("%s %s %s" % (self.prepend_alsvinncli, self.alsvinncli, xmlFilename))
        else:
            os.system("{prepend} mpirun -np {mpi_threads} {alsvinncli} --multi-x {multix} --multi-y {multiy} --multi-z {multiz} {xml}".format(
                mpi_threads=mpi_threads, multix=multix, multiy=multiy,multiz=multiz,xml=xmlFilename, alsvinncli=self.alsvinncli,
            prepend = self.prepend_alsvinncli)
            )


    def __make_string(self, dictionary, keys):
        for key in keys:
            value = dictionary[key]
            formatString = "{} {} {}"

            dictionary[key] = formatString.format(value[0], value[1], value[2])

    def get_line_segment(self):
        min_x = self.fvmSettings["grid"]["lowerCorner"][0]
        max_x = self.fvmSettings["grid"]["upperCorner"][0]
        nx = self.fvmSettings["grid"]["dimension"][0]

        x = numpy.linspace(min_x, max_x, nx)
        x+= 0.5*(x[1]-x[0])

        return x

    def get_2d_grid(self):
        min_x = self.fvmSettings["grid"]["lowerCorner"][0]
        max_x = self.fvmSettings["grid"]["upperCorner"][0]
        nx = self.fvmSettings["grid"]["dimension"][0]

        min_y = self.fvmSettings["grid"]["lowerCorner"][1]
        max_y = self.fvmSettings["grid"]["upperCorner"][1]
        ny = self.fvmSettings["grid"]["dimension"][1]

        x, y = numpy.mgrid[min_x:max_x:nx*1j, min_y:max_y:ny*1j]


        return x, y
    def get_dimension(self):
        dimension = 0
        for n in range(3):
            if self.fvmSettings["grid"]["dimension"][n] > 1:
                dimension+=1
            else:
                break
        return dimension

    def get_data(self, variable, timestep):
        basename = self.fvmSettings["writer"]["basename"]
        type = self.fvmSettings["writer"]["type"]

        if type == "netcdf":
            append = "nc"

        else:
            raise Exception("unknown file format " + type)

        filename = "{basename}_{timestep}.{type}".format(basename=basename, timestep=timestep, type=append)


        if type == "netcdf":
            with netCDF4.Dataset(filename) as f:
                dimension = self.get_dimension()
                if dimension == 1:
                    data = f.variables[variable][:,0,0]
                elif dimension == 2:
                    data = f.variables[variable][:,:,0]
                else:
                    data = f.variables[variable][:,:,:]

        return data

    def get_time(self, timestep):
        basename = self.fvmSettings["writer"]["basename"]
        type = self.fvmSettings["writer"]["type"]

        if type == "netcdf":
            append = "nc"

        else:
            raise Exception("unknown file format " + type)

        filename = "{basename}_{timestep}.{type}".format(basename=basename, timestep=timestep, type=append)
        if type == "netcdf":
            with netCDF4.Dataset(filename) as f:
                return f.variables['time'][0]

    def get_name(self):
        return self.fvmSettings["name"]

    def plot(self, variable, timestep):
        data = self.get_data(variable, timestep)
        dimension = self.get_dimension()
        time = self.get_time(timestep)
        matplotlib.pyplot.title("{name}, plotting {variable} at $T={time}$ ($ts={ts}$)".format(name=self.get_name(), variable=variable,
                                                                                 time=time, ts=timestep))
        if dimension == 1:
            x = self.get_line_segment()
            matplotlib.pyplot.plot(x, data)
            matplotlib.pyplot.xlabel("$x$")
            matplotlib.pyplot.ylabel('${variable}(x,{time})$'.format(variable=variable, time=time))
        elif dimension == 2:
            x,y = self.get_2d_grid()
            matplotlib.pyplot.pcolormesh(x, y, data)
            matplotlib.pyplot.colorbar()
            matplotlib.pyplot.xlabel("$x$")
            matplotlib.pyplot.ylabel('$y$')
        else:
            raise Exception("We do not support 3d yet")
        matplotlib.pyplot.show()

    def __readValues(self, output, document):

        for child in document.childNodes:

            if child.nodeType == child.TEXT_NODE:
                continue
            key = child.tagName
            values = []
            names = []
            textNode = False
            for possibleValue in child.childNodes:
                if possibleValue.nodeType == child.TEXT_NODE:
                    if possibleValue.nodeValue.strip() != '':
                        values.append(possibleValue)
                        textNode = True
                else:
                    values.append(possibleValue)
                    names.append(possibleValue.tagName)



            if textNode:
                output[key] = values[0].nodeValue.strip()
            elif len(set(names)) == len(values):
                output[key] = {}
                self.__readValues(output[key], child)
            else:
                output[key] = []
                for value in values:
                    output[key].append(value.firstChild.nodeValue)


def run(name="alsvinn_experiment", equation=None,
        lower_corner=None,
        upper_corner=None,
        dimension=None,
        flux=None,
        T=None,
        boundary=None,
        reconstruction=None,
        cfl=None,
        integrator=None,
        initial_parameters=None,
        number_of_saves=None,
        initial_data_file=None,
        initial_data_script=None,
        base_xml=None,
        equation_parameters=None,
        platform=None,
        alsvinncli=alsvinn.config.ALSVINNCLI_PATH,
        prepend_alsvinncli=''
        ):


    if base_xml is not None:
        alsvinn_object = Alsvinn(base_xml, alsvinncli=alsvinncli, prepend_alsvinncli=prepend_alsvinncli)
    else:
        alsvinn_object = Alsvinn(alsvinncli=alsvinncli, prepend_alsvinncli=prepend_alsvinncli)
        if lower_corner is None:
            lower_corner = [-5, 0, 0]
        if upper_corner is None:
            upper_corner = [5, 0, 0]
        if dimension is None:
            dimension = [128, 1, 1]
        if flux is None:
            flux = "hll3"
        if T is None:
            T = 1.3
        if boundary is None:
            boundary = "neumann"
        if reconstruction is None:
            reconstruction = "weno2"
        if cfl is None:
            cfl = "auto"
        if integrator is None:
            integrator = "auto"

        if initial_parameters is None:
            initial_parameters = {}

        if number_of_saves is None:
            number_of_saves = 1

        if initial_data_file is None:
            initial_data_file = "%s/sodshocktube/sodshocktube.py" % alsvinn.config.ALSVINN_EXAMPLES_PATH

        if equation_parameters is None:
            equation_parameters = {"gamma": 1.4}

        if platform is None:
            platform = "cpu"
    if name is not None:
        alsvinn_object.set_fvm_value("name", name)
    if equation is not None:
        alsvinn_object.set_fvm_value("equation", equation)
    if flux is not None:
        alsvinn_object.set_fvm_value("flux", flux)
    if T is not None:
        alsvinn_object.set_fvm_value("endTime", T)
    if boundary is not None:
        alsvinn_object.set_fvm_value("boundary", boundary)
    if reconstruction is not None:

        alsvinn_object.set_fvm_value("reconstruction", reconstruction)
    if cfl is not None:
        alsvinn_object.set_fvm_value("cfl", cfl)
    if platform is not None:
        alsvinn_object.set_fvm_value("platform", platform)

    if integrator is not None:
        alsvinn_object.set_fvm_value("integrator", integrator)
    if number_of_saves is not None:
        alsvinn_object.set_fvm_writer("netcdf", name, number_of_saves)

    if dimension is not None and upper_corner is not None and lower_corner is not None:
        alsvinn_object.set_cartesian_grid(lower_corner, upper_corner, dimension)
    elif dimension is not None:
        alsvinn_object.set_dimension(dimension)
    elif lower_corner is not None:
        alsvinn_object.set_lower_corner(lower_corner)
    elif upper_corner is not None:
        alsvinn_object.set_upper_corner(upper_corner)
    if initial_data_script != None:
        initial_data_file = name + ".py"
        with open(initial_data_file, "w") as f:
            f.write(initial_data_script)
        alsvinn_object.set_initial_data(initial_data_file, initial_parameters)
    elif initial_data_file is not None:
        shutil.copyfile(initial_data_file, name+'.py')
        initial_data_file=name+'.py'
        alsvinn_object.set_initial_data(initial_data_file, initial_parameters)
    if equation_parameters is not None:
        alsvinn_object.set_equation_parameters(equation_parameters)
    alsvinn_object.run()
    return alsvinn_object




