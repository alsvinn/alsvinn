#!/bin/env python
import sys
from mpi4py import MPI
import os
import os.path

def runAlsvinn(alsvinnpath, alsmlmc, work, relaxTime):
    if len(work) == 1:
        commandToRun = "%s %s" % (alsvinnpath, work[0])
    else:
        commandToRun = "%s -f %s -c %s -s %f" % (alsmlmc, work[0], work[1], relaxTime)
    print("Running\n\t%s" % commandToRun)
    os.system(commandToRun)

if len(sys.argv) <= 3 or len(sys.argv) > 4:
    print("Wrong number of arguments supplied")
    print("Usage:")
    print("\tpython %s <path to alsvinncli> <path to alsmlmc> (optional: stabilization time, default=0.01)" % sys.argv[0])
    sys.exit(1)

alsvinnExecutable = os.path.abspath(sys.argv[1])
alsmlmcExecutable = os.path.abspath(sys.argv[2])
relaxTime = 0.01 if len(sys.argv) == 3 else float(sys.argv[3])
print("Using relaxation time: %f" % relaxTime)
import nodes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import time

start = time.time()
import os.path
import os

currentDir = os.path.abspath(os.curdir)

for work in nodes.workLists[rank]:
    basepath = os.path.dirname(work[0])
    os.chdir(basepath)
    runAlsvinn(alsvinnExecutable, alsmlmcExecutable, work, relaxTime)

end = time.time()
os.chdir(currentDir)
with open("node_info_%d.txt" % rank, "w") as infoFile:
    infoFile.write("Time spent %f" % (end - start))
    


    
