import generate
import xml.dom.minidom
import os.path
import sys

class GenerateMLMC(object):
    def __init__(self, basefile, outdir, M, numberOfRandomVariables, L):
        self.basefile = os.path.abspath(basefile)
        self.baseinputpath = os.path.dirname(self.basefile) + os.sep
        self.outdir = outdir
        self.M = M
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


    def generateLevels(self):

        for level in range(L):
            levelDir = os.path.join(self.outdir, "level_%d" % level)
            if not os.path.exists(levelDir):
                os.mkdir(levelDir)

            resolution = [x/(2**level) for x in self.resolution]

            xmlFile = self.__makeBaseXML(resolution, level, levelDir)
            print "xmlFile = %s" % xmlFile
            gen = generate.Generate(xmlFile, levelDir, (4**level) * self.M, self.numberOfRandomVariables)

            xmlFiles = gen.generateInputFiles()

            for xmlFile in xmlFiles:
                self.makeCoarseXMLFile(xmlFile, [x/2 for x in resolution])


if __name__ == "__main__":
     if len(sys.argv) != 6:
        print("Not enough parameters supplied")
        print("Usage")
        print("\tpython %s <number of samples> <number of levels> <basefile.xml> <output directory> <number of nodes>" % sys.argv[0])
        sys.exit(1)
    
    
     M = int(sys.argv[1])
     L = int(sys.argv[2])
     basefile = sys.argv[3]
     outdir = sys.argv[4]
     numberOfNodes = int(sys.argv[5])
     
     numberOfRandomVariables = 10
     
     gen = GenerateMLMC(basefile, outdir, M, numberOfRandomVariables, L)
     gen.generateLevels()
     