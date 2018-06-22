import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt

class PlotKH(object):

    def __init__(self, parameters):
        self.ts=0

        self.mpi_rank = 0
        self.mpi_size = 0
        self.sample = 'sample_0'
        if 'mpi_rank' in parameters:
            print (parameters['mpi_rank'])
            self.mpi_rank = int(parameters['mpi_rank'])
            self.mpi_size = int(parameters['mpi_size'])
        if 'group_names' in parameters:
            group_names = parameters['group_names'].split()
            self.sample = group_names[int(parameters['group_index'])]

        print("Constructed")
        

    def write(self, conserved, extra):
        print("plotting")
        plt.pcolormesh(conserved['rho'][:,:,0],vmin=0.9,vmax=2.1)
        plt.colorbar()
        plt.savefig('kh_%s_%d_%d.png' % (self.sample, self.mpi_rank, self.ts))
        plt.close("all")
        self.ts += 1
        print("Plotted")


    def finalize(self):
        pass

    
