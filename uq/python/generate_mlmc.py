import generate
import xml.dom.minidom
import os.path
import sys
import heapq

class GenerateMLMC(object):
    def __init__(self, basefile, outdir, M, numberOfRandomVariables, L, m):
        self.basefile = os.path.abspath(basefile)
        self.baseinputpath = os.path.dirname(self.basefile) + os.sep
        self.outdir = outdir
        self.M = M
        self.m = m
        self.numberOfRandomVariables = numberOfRandomVariables
        self.L = L

        self.__readResolution()

    def __readResolution(self):
        document = xml.dom.minidom.parse(self.basefile)
        resolutionAsString = str(document.getElementsByTagName("fvm")[0].getElementsByTagName("grid")[0].getElementsByTagName("dimension")[0].firstChild.nodeValue).strip()
        self.resolution = [int(x) for x in resolutionAsString.split()]
        print self.resolution

    def makeTemporaryXMLFormat(self, l, outdir):
        return os.path.join(outdir, "%s_%d.xml" % (os.path.basename(os.path.splitext(self.basefile)[0]), l))

    def __makeBaseXML(self, resolution, l, outdir):
        filename = self.makeTemporaryXMLFormat(l, outdir)
        document = xml.dom.minidom.parse(self.basefile)

        resolutionAsString = "%d %d %d" % (resolution[0], resolution[1], resolution[2])

        document.getElementsByTagName("fvm")[0].getElementsByTagName("grid")[0].getElementsByTagName("dimension")[0].firstChild.nodeValue = resolutionAsString

        with open(filename, "w") as outfile:
            document.writexml(outfile)

        pythonFileIn = os.path.join(self.baseinputpath, (document.getElementsByTagName("fvm")[0].getElementsByTagName("initialData")[0].getElementsByTagName("python")[0].firstChild.nodeValue).strip())

        pythonFileOut = os.path.join(outdir, os.path.basename(pythonFileIn))
        print ("pythonfileOUt = %s" % pythonFileOut)
        with open(pythonFileIn, "r") as pythonIn:
            with open(pythonFileOut, "w") as pythonOut:
                pythonOut.write(pythonIn.read())
        return filename

    def makeCoarseXMLFile(self, xmlFile, resolution):
        document = xml.dom.minidom.parse(xmlFile)
        outfilename = os.path.splitext(xmlFile)[0] + "_coarse" + ".xml"

        resolutionAsString = "%d %d %d" % (resolution[0], resolution[1], resolution[2])

        document.getElementsByTagName("fvm")[0].getElementsByTagName("grid")[0].getElementsByTagName("dimension")[0].firstChild.nodeValue = resolutionAsString

        baseout = str(document.getElementsByTagName("fvm")[0].getElementsByTagName("writer")[0].getElementsByTagName("basename")[0].firstChild.nodeValue).strip()

        baseout = baseout + "_coarse"
      


        document.getElementsByTagName("fvm")[0].getElementsByTagName("writer")[0].getElementsByTagName("basename")[0].firstChild.nodeValue = baseout

        with open(outfilename, "w") as outfile:
            document.writexml(outfile)
        return outfilename


    def generateLevels(self):
        self.xmlPairs = []
        self.resolutions = []
        self.xmlPairs.append([])
        self.resolutions = [[] for x in range(self.L + 1)]
        for level in range(0, self.L + 1):
            self.xmlPairs.append([])
            levelDir = os.path.join(self.outdir, "level_%d" % level)
            if not os.path.exists(levelDir):
                os.mkdir(levelDir)

            resolution = [max(x/(2**(self.L-level)),1) for x in self.resolution]

            self.resolutions[level]= resolution
            xmlFile = self.__makeBaseXML(resolution, level, levelDir)
            print "xmlFile = %s" % xmlFile
            print "blah = %d" % self.M

            numberOfSamples = (4**(self.L-level)) * self.M if level > 0 else self.m
            gen = generate.Generate(xmlFile, levelDir, numberOfSamples, self.numberOfRandomVariables)

            xmlFiles = gen.generateInputFiles()

            for xmlFile in xmlFiles:
                xmlFile = os.path.abspath(xmlFile)
                if level > 0:
                    coarseFile = self.makeCoarseXMLFile(xmlFile, [max(x/2, 1) for x in resolution])
                    self.xmlPairs[level].append([xmlFile, coarseFile])
                else:
                    self.xmlPairs[level].append([xmlFile])


    def setPlatform(self, xmlFile, platform):
         document = xml.dom.minidom.parse(xmlFile)
         document.getElementsByTagName("fvm")[0].getElementsByTagName("platform")[0].firstChild.nodeValue = platform
         with open(xmlFile, "w") as outfile:
             document.writexml(outfile)
         
    def distributeWork(self, numberOfNodesToUse, useGPU=True):
        workList = [[] for node in range(numberOfNodesToUse)]
        loadList = []
        for node in range(numberOfNodesToUse):
            heapq.heappush(loadList, (0, node))
        
        for level in range(self.L, -1, -1):
            resolution = self.resolutions[level]

            numberOfCells = resolution[0]*resolution[1]*resolution[2]

            if numberOfCells > 1024:
                useGPUOnLevel = useGPU
            else:
                useGPUOnLevel = False

            for xmlPair in self.xmlPairs[level]:
                nextNode = heapq.heappop(loadList)

                if len(xmlPair) > 1:
                    if useGPUOnLevel:
                        self.setPlatform(xmlPair[0], "cuda")
                        
                workList[nextNode[1]].append(xmlPair)
                # Add that work also depends on the CFL condition
                additionalWork = numberOfCells * resolution[0]
                if not useGPUOnLevel and len(xmlPair) > 1:
                    additionalWork *= 2
                newWork = nextNode[0] +additionalWork
                heapq.heappush(loadList, (newWork, nextNode[1]))
        with open(os.path.join(self.outdir, "nodes.py"), "w") as nodeFile:
            nodeFile.write("workLists = %s\n" % workList)
                



if __name__ == "__main__":
     if len(sys.argv) != 7:
        print("Not enough parameters supplied")
        print("Usage")
        print("\tpython %s <number of samples on finest level> <number of levels> <number of samples on coarsest level> <basefile.xml> <output directory> <number of nodes>" % sys.argv[0])
        sys.exit(1)
    
    
     M = int(sys.argv[1])
     m = int(sys.argv[3])
     L = int(sys.argv[2])
     basefile = sys.argv[4]
     outdir = sys.argv[5]
     numberOfNodes = int(sys.argv[6])
     
     numberOfRandomVariables = 10
     
     gen = GenerateMLMC(basefile, outdir, M, numberOfRandomVariables, L, m)
     gen.generateLevels()
     gen.distributeWork(numberOfNodes)
