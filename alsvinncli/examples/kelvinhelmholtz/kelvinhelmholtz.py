
GAMMA=5.0/3.0
import random
# instability
perturbation = 0
if y < 0.25 + perturbation or y > 0.75 + perturbation:
    rho = 1
    ux = 0.5
    uy = 0
    p = 2.5
else:
    rho = 2
    ux = -0.5
    uy = 0
    p = 2.5


