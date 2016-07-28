"""
Small program to insert a new class into alsvinn/alsfvm

This script will create the necessary folders and create the boilerplate
code (namespace, includes, etc.). It will also add the files to git.
"""

import os
import os.path
import sys

"""
Adds the current file to version control (does not commit!)
"""
def addToGit(filename):
    print "Adding %s to git repo" % filename
    os.system("git add %s" %filename)
    
def directoryExists(directoryName):
    if (not os.path.isdir(directoryName)) and (os.path.exists(directoryName)):
        raise Exception("File %s exists but is not a directory" % directoryName)
    return os.path.isdir(directoryName)

def createDirectory(directory):
    os.mkdir(directory)

def writeToFile(filename, content):
    if os.path.exists(filename):
        raise Exception("File %s already exists!" % filename)
    
    print "Writing to %s" % filename
    splitName = filename.split("/")
    fullPath = ""
    for name in splitName[0:-1]:
        fullPath += name +  "/"
        if not directoryExists(fullPath):
            createDirectory(fullPath)
    with open(filename, "w") as file:
        file.write(content)
    
"""
Gets the Hpp filename we will use

qualifedClassname is on the form foo::bar::Classname
"""
def getHppFilename(qualifiedClassname):
    splitName = qualifiedClassname.split("::")
    path = "include/"
    for name in splitName:
        path += "%s/" % name

    path = path[:-1]
    path += ".hpp"
    return path

"""
Gets the Hpp filename we will use

qualifedClassname is on the form foo::bar::Classname
"""
def getCppFilename(qualifiedClassname):
    splitName = qualifiedClassname.split("::")
    path = "src/"
    # We don't include the alsfvm part in the start
    for name in splitName[1:]:
        path += "%s/" % name

    path = path[:-1]
    path += ".cpp"
    return path
        
def createClassTextHpp(qualifiedClassname):
    classText = "#pragma once" + "\n\n";
    splitName = qualifiedClassname.split("::")

    # Create namespaces
    for name in splitName[0:-1]:
        classText += "namespace %s { " % name
    classText += "\n"
    classText += "\n"
    # Add classname
    classText += "    class %s {" % splitName[-1]

    # Add public at start
    classText += "\n"
    classText += "    public:" + "\n"
    classText += "\n"
    classText += "    };" + "\n"
    # close namespaces
    for name in reversed(splitName[0:-1]):
        classText += "} // namespace %s" % name
        classText += "\n"

    return classText


def createClassTextCpp(qualifiedClassname):
    text = "#include \"%s\"" % getHppFilename(qualifiedClassname).replace("include/", "")
    text += "\n" + "\n"
    splitName = qualifiedClassname.split("::")
    for name in splitName[0:-1]:
        text += "namespace %s { " % name

    text += "\n" + "\n"

    # close namespaces
    for name in splitName[0:-1]:
        text += "}"
        text += "\n"
    
    return text
    
if __name__ == "__main__":
    className = sys.argv[1]
    if not className.startswith("alsfvm"):
        className = "alsfvm::" + className
        
    print "Creating class %s" % className
    classTextHpp = createClassTextHpp(className)
    classTextCpp = createClassTextCpp(className)

    hppFile = getHppFilename(className)
    cppFile = getCppFilename(className)

    writeToFile(hppFile, classTextHpp)
    writeToFile(cppFile, classTextCpp)
    addToGit(hppFile)
    addToGit(cppFile)
