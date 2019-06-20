

perturbation = 0.01

perturbation_upper = perturbation*cos(pi*a*x+pi*a)
perturbation_lower = perturbation*cos(pi*a*x+pi*a)

if y < 0.25 + perturbation_lower or y > 0.75 + perturbation_upper:
    rho = 1
    ux = 0.5
    uy = 0
    p = 2.5
else:
    rho = 2
    ux = -0.5
    uy = 0
    p = 2.5


