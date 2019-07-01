"""
This module file is solely for running and plotting alsvinn results
it does not contain any alsvinn logic. You can obtain all the results by just running the alsvinncli executable

This file is meant for scripting purposes (integrates well with jupyter notebook for instance)

"""
from alsvinn.config import *
import xml.dom.minidom
import numpy
import os
import dicttoxml
import copy
import shutil
import netCDF4
import matplotlib
import matplotlib.pyplot
import subprocess
import tempfile
import re

class Alsvinn(object):
    def __init__(self, xml_file=None, configuration=None,
                 alsvinncli=ALSVINNCLI_PATH,
                 prepend_alsvinncli='',
                 alsuqcli=ALSUQCLI_PATH,
                 omp_num_threads=None, data_path=''):
        self.settings = {"fvm" : {}, "uq" : {}}
        self.fvmSettings = self.settings["fvm"]
        self.uqSettings=self.settings["uq"]
        self.data_path = data_path
        if xml_file != None:
            self.read_xml(xml_file)
        
        elif configuration != None:
            self.fvmSettings = configuration
        self.alsvinncli = alsvinncli
        self.prepend_alsvinncli = prepend_alsvinncli
        self.omp_num_threads=omp_num_threads
        self.alsuqcli = alsuqcli




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

    def set_fvm_writer(self, type, basename, number_of_saves, write_initial_timestep):
        self.fvmSettings["writer"] = {}
        self.fvmSettings["writer"]["type"] = type
        self.fvmSettings["writer"]["basename"] = basename
        self.fvmSettings["writer"]["numberOfSaves"] = number_of_saves
        self.fvmSettings["writer"]["writeInitialTimestep"] = int(write_initial_timestep)
        

    def set_cartesian_grid(self, lower_corner, upper_corner, dimensions):
        self.fvmSettings["grid"] = {"lowerCorner" : lower_corner,
                                    "upperCorner" : upper_corner,
                                    "dimension" : dimensions}


    def read_xml(self, xml_file):
        self.source_xml_path=xml_file
        document = xml.dom.minidom.parse(xml_file)
        configDocument = document.getElementsByTagName("config")[0]
        fvmDocument = configDocument.getElementsByTagName("fvm")[0]
        if len(configDocument.getElementsByTagName("uq")) > 0:
            uqDocument = configDocument.getElementsByTagName("uq")[0]
            self.__readValues(self.uqSettings, uqDocument)
            
            if not self.uqSettings['parameters']:
                self.uqSettings['parameters'] = self.fvmSettings['initialData']['parameters']
                self.uqSettings['parameters']['parameter']['type'] = 'uniform'
                del self.uqSettings['parameters']['values']

                
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

    def run(self, multix=1,multiy=1,multiz=1, uq=False, multiSample=1):
        mpi_threads = multiSample*multix*multiy*multiz
        xmlFilename = self.fvmSettings["name"] + ".xml"

        settings = copy.deepcopy(self.settings)
        self.__make_string(settings["fvm"]["grid"], ["lowerCorner", "upperCorner", "dimension"])

        newPythonFile = self.get_name() + '.py'

        if settings["fvm"]["initialData"]["python"] != newPythonFile:
            shutil.copyfile(os.path.join(os.path.dirname(self.source_xml_path), settings["fvm"]["initialData"]["python"]), newPythonFile)
            settings["fvm"]["initialData"]["python"] = newPythonFile

        xmlObject = dicttoxml.dicttoxml( settings, custom_root='config', attr_type=False, item_func=self.__xml_item_func)

        xmlDocument = xml.dom.minidom.parseString(xmlObject)
        with open (xmlFilename, "w") as f:
            f.write(xmlDocument.toprettyxml())

        runCommand = self.alsvinncli

        if uq:
            runCommand = self.alsuqcli

        if mpi_threads == 1:
            commandArray = [str(runCommand), str(xmlFilename)]
        else:

            commandArray = [ 'mpirun', '-np', str(mpi_threads), runCommand,
                            '--multi-x', str(multix), '--multi-y', str(multiy), '--multi-z', str(multiz)]
            if uq:
                commandArray.append('--multi-sample')
                commandArray.append(str(multiSample))
            commandArray.append(xmlFilename)

        if self.prepend_alsvinncli and self.prepend_alsvinncli.strip() != '':
            commandArray  = self.prepend_alsvinncli.split() + commandArray
        env = copy.deepcopy(os.environ)
        if self.omp_num_threads:
            env['OMP_NUM_THREADS'] = str(self.omp_num_threads)
        commandObject = subprocess.Popen(commandArray, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        output, error = commandObject.communicate()
        returnCode = commandObject.returncode

        if returnCode != 0:
            errorMessage = ("Error running alsvinncli.\n\n The command used was\n\n\t{command}\n\nThe output was:\n" \
            + "\n----------------\n{output}\n----------------\n\nThe error output was:\n\n----------------\n{error}\n\n----------------\n\nAlso check the log files 'alsvinncli_mpi_log_<n>.txt'").format(
                command=' '.join(commandArray), output=str(output).replace('\\n','\n'), error=str(error).replace('\\n','\n')
            )
            raise Exception(errorMessage)



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

    def get_data(self, variable, timestep, sample=0, statistics=None):
        if statistics is None:
            basename = self.fvmSettings["writer"]["basename"]
            type = self.fvmSettings["writer"]["type"]
        else:

            # Loop through uq stats to find statistics:
            for s in self.uqSettings['stats']:

                if isinstance(s, str):
                    if len(s.strip()) == 0:
                        continue
                    s = self.uqSettings['stats'][s]

                if statistics == 'mean' or statistics == 'variance':
                    
                    if s['name'] == 'meanvar':

                        basename = s['writer']['basename']
                        type = s['writer']['type']
                        break
                elif re.match(r'm(\d+)', statistics):
                    match = re.match(r'm(\d+)', statistics)
                    p_wanted = int(match.group(1))

                    if 'p' in s and int(s['p']) == p_wanted:
                        basename = s['writer']['basename']
                        type = s['writer']['type']
                        break
                elif s['name'] == statistics:
                    basename = s['writer']['basename']
                    type = s['writer']['type']
                    break
            else:
                raise Exception(f"Statistics not found: {statistics}")
            
        if type == "netcdf":
            append = "nc"

        else:
            raise Exception("unknown file format " + type)

        if statistics is None:
            filename = "{basename}_{timestep}.{type}".format(basename=basename, timestep=timestep, type=append)
        else:
            filename = "{basename}_{statistics}_{timestep}.{type}".\
                format(basename=basename, timestep=timestep, type=append, statistics=statistics)

        filename = os.path.join(self.data_path, filename)

        if type == "netcdf":
            with netCDF4.Dataset(filename) as f:
                if not variable in f.variables.keys() or sample>0:
                    variable = 'sample_{sample}_{variable}'.format(sample=sample, variable=variable)
                data = f.variables[variable]
                shape = data.shape
                dimension = 0
                for n in shape:
                    if n>1:
                        dimension+=1

                if dimension == 1:
                    data = data[:,0,0]
                elif dimension == 2:
                    data = data[:,:,0]
                else:
                    data = data[:,:,:]

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

    def plot(self, variable, timestep, sample=0, statistics=None):
        data = self.get_data(variable, timestep, sample, statistics)
        dimension = self.get_dimension()
        time = self.get_time(timestep)

        if statistics is None:
            matplotlib.pyplot.title("{name}, plotting {variable} at $T={time}$ ($ts={ts}$)".format(name=self.get_name(), variable=variable,
                                                                                 time=time, ts=timestep))
        else:
            matplotlib.pyplot.title(
                "{name}, plotting {statistics} of  {variable} at $T={time}$ ($ts={ts}$)".format(statistics=statistics, name=self.get_name(), variable=variable,
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

    def __isPureTextNode(self, node):

        for k in node.childNodes:
            if k.nodeType != node.TEXT_NODE:
                return False

        return True

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
                    if self.__isPureTextNode(value):
                        output[key].append(value.firstChild.nodeValue)
                    else:
                        new_dict = {}
                        self.__readValues(new_dict, value)
                        output[key].append(new_dict)


    def set_uq_value(self, key, value):
        self.uqSettings[key] = value

    def add_statistics(self, statisticsName, writer, basename, numberOfSaves, writeInitialTimestep):
        if 'stats' not in self.uqSettings.keys():
            self.uqSettings['stats']  = []
        elif type(self.uqSettings['stats']) is not list:
            self.uqSettings['stats'] = [self.uqSettings['statistics']['stat']]
        self.uqSettings['stats'].append({'name' : statisticsName,
                                                'numberOfSaves' : numberOfSaves,
                                                "writeInitialTimestep" : int(writeInitialTimestep),
                                                'writer':{
                                                    'type' : writer,
                                                    'basename' : basename
                                                }})

    def add_functional(self, functionalOptions):
        if 'functionals' not in self.fvmSettings.keys():
            self.fvmSettings['functionals']  = []
        elif type(self.fvmSettings['functionals']) is not list:
            self.fvmSettings['functionals'] = [self.fvmSettings['functionals']['functional']]
        self.fvmSettings['functionals'].append(functionalOptions)

    def set_distribution(self, distribution):
        self.uqSettings['parameter']['parameter']['type']=distribution

    def set_diffusion(self, operator, reconstruction):
        self.fvmSettings['diffusion'] = {'name' : operator,
            'reconstruction' : reconstruction}


def run(name=None, equation=None,
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
        alsvinncli=ALSVINNCLI_PATH,
        alsuqcli=ALSUQCLI_PATH,
        prepend_alsvinncli='',
        omp_num_threads=None,
        multix=1,
        multiy=1,
        multiz=1,
        uq=False,
        sampleStart=None,
        multiSample=1,
        statistics=None,
        samples=None,
        generator=None,
        diffusion_operator=None,
        diffusion_reconstruction=None,
        functionals=None,
        write_initial_timestep=True
        ):

    initial_data_file_is_temporary = False


    if base_xml is not None:
        alsvinn_object = Alsvinn(base_xml, alsvinncli=alsvinncli, alsuqcli=alsuqcli,prepend_alsvinncli=prepend_alsvinncli, omp_num_threads=omp_num_threads)
    else:

        alsvinn_object = Alsvinn(alsvinncli=alsvinncli, alsuqcli=alsuqcli, prepend_alsvinncli=prepend_alsvinncli, omp_num_threads=omp_num_threads)
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
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix='.py') as f: 
                initial_data_file = f.name
                initial_data_file_is_temporary = True

                f.write("""
# Sod shock tube
if x < 0.0:
    rho = 1.
    ux = 0.
    p = 1.0
else:
    rho = .125
    ux = 0.
    p = 0.1
"""
                        )

        if equation_parameters is None:
            equation_parameters = {"gamma": 1.4}

        if equation is None:
            equation = "euler1"
        if platform is None:
            platform = "cpu"
        if samples is None:
            samples = 1
        if generator is None:
            generator = 'auto'
        if statistics is None:
            statistics = ['meanvar']
        if name is None:
            name = 'alsvinn_experiment'
        if diffusion_operator is None:
            diffusion_operator = 'none'
        if diffusion_reconstruction is None:
            diffusion_reconstruction = 'none'
    if name is not None:
        alsvinn_object.set_fvm_value("name", name)
    else:
        name = alsvinn_object.fvmSettings['name']
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
        alsvinn_object.set_fvm_writer("netcdf", name, number_of_saves, write_initial_timestep)

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

    if statistics is not None:

        if number_of_saves is None:
            number_of_saves = 1
        alsvinn_object.set_uq_value('stats',[])
        for stat in statistics:
            alsvinn_object.add_statistics(stat, 'netcdf', name, number_of_saves, write_initial_timestep)
    if functionals is not None:
        if number_of_saves is None:
            number_of_saves = 1
        alsvinn_object.set_fvm_value('functionals',[])
        for functional in functionals:
            alsvinn_object.add_functional(functional)
    if samples is not None:
        alsvinn_object.set_uq_value('samples', str(samples))
    if generator is not None:
        alsvinn_object.set_uq_value('generator', generator)

    if sampleStart is not None:
        alsvinn_object.set_uq_value('sampleStart', sampleStart)

    if diffusion_operator is not None:
        alsvinn_object.set_diffusion(diffusion_operator, diffusion_reconstruction)
    alsvinn_object.run(multix=multix, multiy=multiy,multiz=multiz, uq=uq, multiSample=multiSample)

    if initial_data_file_is_temporary:
        os.remove(initial_data_file)
    return alsvinn_object




