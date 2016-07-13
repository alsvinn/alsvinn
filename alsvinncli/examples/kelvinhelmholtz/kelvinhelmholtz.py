
GAMMA=1.4
import random
# instability

if "has_random_variables" in locals() and has_random_variables:
	N = len(a1)
	perturbation = 0.01
	normalization1 = sum(a1)
	if abs(normalization1) < 1e-10:
		normalization1 = 1
	normalization2 = sum(a2)
	if abs(normalization2) < 1e-10:
		normalization2 = 1

	perturbation_upper = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(x+b1[i])) for i in range(len(a1))])/normalization1
	perturbation_lower = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(x+b2[i])) for i in range(len(a2))])/normalization2
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


