"""
Computes the convergence in 1D for the sod shock tube problem

The reference solution is generated with alsvid at a resolution of $2^{15}$ cells.
"""




import alsvinn
import alsvinn.comparison
basefile = "sodshock.xml"
reference_solution = ("sodshock_alsvid.h5", "alsvid")

configurations = alsvinn.ALL_CONFIGURATIONS

resolutions = [[2**k, 1, 1] for k in range(4,15)]
required_convergence_rate = 0.8

data = alsvinn.comparison.convergence_compare(basefile, reference_solution, configurations, resolutions, required_convergence_rate)

report = alsvinn.Report("sodshock", "Sod Shock tube", __doc__)

report.add_convergence_plot(resolutions, data)
report.write("sodshock.tex")





