"""
Computes the convergence in 1D for the Sod shock tube problem for the compressible Euler equations

The initial data is given as
\\[\\mathbf{U}=\\begin{pmatrix}\\rho& u& p\\end{pmatrix}=\\begin{pmatrix}3&0&3\\end{pmatrix}\\mathbf{1}_{[0,1]}(x)+\\begin{pmatrix}1&0&1\\end{pmatrix}\\mathbf{1}_{[1,2]}(x)\\]

The reference solution is generated with AlsvidUQ at a resolution of $2^{15}$ cells. We compute up to $t=0.5$.

Because of the discontinuity, we can only expect a convergence rate of $1/2$ for the simplest configurations.
"""




import alsvinn
import alsvinn.data
import alsvinn.comparison
import alsvinn.runner.alsvinn
import numpy

basefile = "sodshock.xml"
reference_solution = ("sodshock_alsvid.h5", "alsvid")
configurations = alsvinn.ALL_CONFIGURATIONS
resolutions = [[2**k, 1, 1] for k in range(4,14)]
required_convergence_rate = 0.5


report = alsvinn.Report("sodshock", "Sod Shock tube", __doc__)


data = alsvinn.comparison.convergence_compare(basefile, reference_solution, configurations, resolutions, required_convergence_rate)



report.add_convergence_plot(resolutions, data)



plot_resolutions = [resolutions[2], resolutions[-1]]

def write_to_report(config, values):
    variable = "rho"
    plot_data = {}
    reference_data_reader = alsvinn.data.make_reader(reference_solution)
    reference_data = reference_data_reader.read_dataset(variable)
    x_reference = numpy.linspace(0, 2, reference_data.shape[0])
    plot_data["Alsvid, %d" % reference_data.shape[0]] = [x_reference, reference_data]
    print("Plotting %s" % config)
    for (n, resolution) in enumerate(plot_resolutions):
        x = numpy.linspace(0, 2, resolution[0])
        data_reader = alsvinn.data.make_reader(values[n])
        data = data_reader.read_dataset(variable)
        plot_data["Alsvinn, %d" % resolution[0]] = [x, data]
    report.add_plot_data(plot_data, "Comparison on %s with Alsvinn configuration %s" % (variable, config))

alsvinn.runner.alsvinn.run_configurations(basefile, configurations, plot_resolutions, write_to_report)

report.write("sodshock.tex")