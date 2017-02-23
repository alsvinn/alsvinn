#!/bin/env python

"""
Makes a LaTeX table of an input file
for easy inclusion into a paper
"""

import sys
import xml.dom.minidom

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n")
        print("\tpython %s <path to xml file>" % sys.argv[0])
        exit(1)

    filename = sys.argv[1]

    f = xml.dom.minidom.parse(filename)

    fields = {"equation":"Equation",
              "reconstruction":"Reconstruction", "integrator":"Time integrator", "cfl": "CFL", "endTime":"$T$","flux":"Numerical flux"}

    fieldNames = ["equation", "flux", "reconstruction", "integrator",  "cfl", "endTime"]
    fvm = f.getElementsByTagName("config")[0].getElementsByTagName('fvm')[0]

    print("\\begin{table}\n")
    sys.stdout.write("\\begin{tabular}{")
    for field in fields.keys():
        sys.stdout.write("l")
    sys.stdout.write("}\n \\toprule\n")
    for (n,field) in enumerate(fieldNames):

        sys.stdout.write ("%s" % fields[field])

        if n < len(fields.keys())-1:
            sys.stdout.write("&")
        else:
            sys.stdout.write("\\\\ \n")

    print("\midrule")
    usedValues = {}

    
    for (n,field) in enumerate(fields.keys()):
        value = fvm.getElementsByTagName(field)[0].firstChild.nodeValue
        usedValues[field] = value
    if usedValues['integrator'] == 'auto':
        if usedValues['reconstruction'] != 'none':
            usedValues['integrator'] = 'rungekutta2'
        else:
            usedValues['integrator'] = 'forwardeuler'
        

    if usedValues['cfl'] == 'auto':
        if usedValues['reconstruction'] != 'none':
            usedValues['cfl'] = '0.475'
        else:
            usedValues['cfl'] = '0.9'

    prettyNames = {'forwardeuler':'Forward-Euler', 'rungekutta2':'SSP RK 2', "burgers":"Burgers", "euler1":"1D Euler", "euler2":"2D Euler", "euler3":"3D Euler"}

    for (n,field) in enumerate(fieldNames):
        value = usedValues[field]
        if value in prettyNames.keys():
            
            value = prettyNames[value]
        elif field == 'reconstruction':
            value = value.upper()
        elif field == 'flux':
            value = value.title()

        sys.stdout.write ("%s" % value)

        if n < len(fields.keys())-1:
            sys.stdout.write("&")
        else:
            sys.stdout.write("\\\\ \n")
    print("\\bottomrule")
    print ("\\end{tabular}")

    print("\\caption{Parameters to the Finite Volume simulation used for the experiment INSERT EXPERIMENT NAME HERE. \\label{tbl:TABLENAME}}")
    print("\\end{table}")
