import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt
import os

def combine_images(images, outfile, horiz=False):
    combine_string = ' '.join(images)
    if horiz:
        command = 'convert +append %s %s'
    else:
        command = 'convert -append %s %s'
    os.system(command % (combine_string, outfile))

def trim_image(image):
    os.system('convert -trim {name} {name}'.format(name=image))

def delete_files(files):
    for f in files:
        os.remove(f)

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

        self.basename = 'kh_{cmap}_{sample}_{x}_{y}_{t}.png'

    def write(self, conserved, grid, time, timesteps):
        nx = int(grid['local_size'][0])
        ny = int(grid['local_size'][1])

        nx_tot = int(grid['global_size'][0])
        ny_tot = int(grid['global_size'][1])

        self.x_position = int(grid['global_position'][0])/nx
        self.y_position = int(grid['global_position'][1])/ny

        self.nodes_x = nx_tot/nx
        self.nodes_y = ny_tot/ny

        rho = reshape(conserved['rho'], (ny, nx))
        for cmap in self.colormaps:

            f = plt.figure(figsize=(int(self.ratio[0]*float(nx)/nx_tot), int(self.ratio[1]*float(ny)/ny_tot)))


            x,y = mgrid[0:1:nx*1j, 0:1:ny*1j]
            plt.pcolormesh(rho, vmin=0.9,vmax=2.1, cmap=cmap)
            plt.axis('off')
            outfile = self.basename.format(cmap=cmap, sample=self.sample, x=self.x_position, y=self.y_position, t=self.ts)
            plt.savefig(outfile, bbox_inches='tight')
            plt.close(f)
            trim_image(outfile)
        self.ts += 1
        del rho
    def afterMPISync(self):
        """
        Combines the images printed to one large image
        """

        if self.y_position == 0 and self.x_position == 0:

            ts = self.ts - 1
            files_to_delete = []
            for cmap in self.colormaps:
                x_files = []
                for i in range(self.nodes_y-1,-1,-1):

                    files = [self.basename.format(cmap=cmap, sample=self.sample, x=j, y=i, t=ts) for j in range(self.nodes_x-1, -1, -1)]
                    output_file = self.basename.format(cmap=cmap, sample=self.sample, x=0, y=i, t=ts)
                    combine_images(files, output_file, horiz=True)
                    x_files.append(output_file)
                    files_to_delete.extend(files)

                output_file = self.basename.format(cmap=cmap, sample=self.sample, x=0, y=0,t=ts)
                combine_images(x_files, output_file, horiz=False)
                files_to_delete.remove(output_file)

                delete_files(files_to_delete)


    def finalize(self):
        pass

    
