#!/bin/env python
import sys
from mpi4py import MPI
import os
import os.path

def runAlsvinn(alsvinnpath, filepath):
    commandToRun = "%s %s" % (alsvinnpath, filepath)
    print("Running\n\t%s" % commandToRun)
    os.system(commandToRun)

if len(sys.argv) != 2:
    print("Wrong number of arguments supplied")
    print("Usage:")
    print("\tpython %s <path to alsvinncli>" % sys.argv[0])
    sys.exit(1)

alsvinnExecutable = os.path.abspath(sys.argv[1])
import nodes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

for work in nodes.workLists[rank]:
    basepath = os.path.dirname(work)
    os.chdir(basepath)
    runAlsvinn(alsvinnExecutable, work)



    
