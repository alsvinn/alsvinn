import sys
import xml.dom.minidom
import os.path
import numpy.random as random

class Generate(object):
    def __init__(self, basefile, outdir, M, numberOfRandomVariables):
        self.basefile = os.path.abspath(basefile)
        self.baseinputpath = os.path.dirname(self.basefile) + os.sep
        print self.baseinputpath
        self.outdir = outdir
        self.M = M
        self.numberOfRandomVariables = numberOfRandomVariables

        self.__readXML()

    def __readXML(self):
        with open(self.basefile, "r") as xmlfile:
            self.xmlfile = xmlfile.read()

        document = xml.dom.minidom.parse(self.basefile)
        self.basename = str(document.getElementsByTagName("fvm")[0].getElementsByTagName("name")[0].firstChild.nodeValue).strip()
        print self.basename

        self.pythonfile = os.path.join(self.baseinputpath, (document.getElementsByTagName("fvm")[0].getElementsByTagName("initialData")[0].getElementsByTagName("python")[0].firstChild.nodeValue).strip())

        self.basepythonfile = os.path.basename(self.pythonfile)
        
    def getOutdirname(self, k):
        return os.path.join(self.outdir, "%s_%d" % (self.basename, k))
    def createOutDir(self, k):
        outdirname = self.getOutdirname(k)
        if not os.path.exists(outdirname):
            os.mkdir(outdirname)
        return outdirname

    def generateInputFiles(self):
        self.xmlfiles = []
        with open(self.pythonfile, "r") as pythonfile:
            basepythonscript = pythonfile.read()

        for k in range(self.M):
            outdirname = self.createOutDir(k)
            outfilename = os.path.join(outdirname, self.basepythonfile)
            randomVars = {}
            randomVars["a1"] = [random.uniform(0, 1) for x in range(self.numberOfRandomVariables)]
            randomVars["b1"] = [random.uniform(0, 1) for x in range(self.numberOfRandomVariables)]

            randomVars["a2"] = [random.uniform(0, 1) for x in range(self.numberOfRandomVariables)]
            randomVars["b2"] = [random.uniform(0, 1) for x in range(self.numberOfRandomVariables)]

            outputscript = "has_random_variables = True\n"
            
            for randomVar in randomVars.keys():
                outputscript += "%s = %s\n\n" % (randomVar, randomVars[randomVar])

            outputscript += basepythonscript
            
            with open(outfilename, "w") as outfile:
                outfile.write(outputscript)
            
            outxmlfilename = os.path.join(outdirname, os.path.basename(self.basefile))
            self.xmlfiles.append(outxmlfilename)
            with open(outxmlfilename, "w") as outxmlfile:
                outxmlfile.write(self.xmlfile)
        return self.xmlfiles

    def distributeWork(self, numberOfNodes):
        with open("nodes.py", "w") as nodefile:
            nodefile.write("workLists = []\n")
            numberOfSamplesPerNode = (self.M + numberOfNodes - 1) / numberOfNodes
            for node in range(numberOfNodes):
                worklist = []
                startSample = numberOfSamplesPerNode*node
                endSample = min(self.M, numberOfSamplesPerNode*(node+1))
                for k in range(startSample, endSample):
                    worklist.append(os.path.abspath(self.xmlfiles[k]))
                nodefile.write("workLists.append(%s)\n\n" % worklist)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Not enough parameters supplied")
        print("Usage")
        print("\tpython %s <number of samples> <basefile.xml> <output directory> <number of nodes>" % sys.argv[0])
        sys.exit(1)
    
    
    M = int(sys.argv[1])
    basefile = sys.argv[2]
    outdir = sys.argv[3]
    numberOfNodes = int(sys.argv[4])
    
    numberOfRandomVariables = 10

    generate = Generate(basefile, outdir, M, numberOfRandomVariables)
    generate.generateInputFiles()
    generate.distributeWork(numberOfNodes)


