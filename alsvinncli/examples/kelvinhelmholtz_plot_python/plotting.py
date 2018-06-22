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
            self.mpi_rank = int(parameters['mpi_rank'])
            self.mpi_size = int(parameters['mpi_size'])
        if 'group_names' in parameters:
            group_names = parameters['group_names'].split()
            self.sample = group_names[int(parameters['group_index'])]

        self.colormaps = ['PuRd', 'jet', 'rainbow', 'viridis']
        self.ratio = [16,9]

    def write(self, conserved, extra, grid):
        nx = int(grid['local_size'][0])
        ny = int(grid['local_size'][1])

        nx_tot = int(grid['global_size'][0])
        ny_tot = int(grid['global_size'][1])
        rho = reshape(conserved['rho'], (ny, nx))
        for cmap in self.colormaps:

            f = plt.figure(figsize=(int(self.ratio[0]*float(nx)/nx_tot), int(self.ratio[1]*float(ny)/ny_tot)))


            x,y = mgrid[0:1:nx*1j, 0:1:ny*1j]
            plt.pcolormesh(rho, vmin=0.9,vmax=2.1, cmap=cmap)
            plt.axis('off')
            plt.savefig('kh_%s_%s_%d_%d.png' % (cmap, self.sample, self.mpi_rank, self.ts), bbox_inches='tight')
            plt.close(f)
        self.ts += 1
        del rho


    def finalize(self):
        pass

    
