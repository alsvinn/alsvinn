
GAMMA=5.0/3.0
import random
# instability
if has_random_variables:
	N = len(a1)
	perturbation = 0.01
	perturbation_upper = perturbation*sum([a1[i]*cos(2*pi*i*(x+b1[i])) for i in range(len(a1))])/N
	perturbation_lower = perturbation*sum([a2[i]*cos(2*pi*i*(x+b2[i])) for i in range(len(a2))])/N
else:
	perturbation_upper = 0
	perturbation_lower = 0

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


