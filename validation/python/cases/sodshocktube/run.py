"""
Computes the convergence
"""
basefile = "sodshock.xml"
reference_solution = ("sodshock_alsvid.h5", "alsvid")

configurations = alsvinn.ALL_CONFIGURATIONS

resolutions = [2**k for k in range(4,15)]
required_convergence_rate = 0.8
convergence_compare(basefile, reference_solution, configurations, resolutions, required_convergence_rate)




