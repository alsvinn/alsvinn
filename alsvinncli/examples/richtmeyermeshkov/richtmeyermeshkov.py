epsilon = 1e-2
xc = x - 0.5
yc = y - 0.5
phi = atan(xc/yc) if abs(yc) > 0 else 0

if "has_random_variables" in locals() and has_random_variables:
    N = len(a1)
    perturbation = sum([a1[n] * cos(phi+b1[n]) for n in xrange(N)])
else:
    perturbation = 0

r = sqrt((x-0.5)**2+(y-0.5)**2)
if r < 0.1:
    p = 20
else:
    p = 1

if r < 0.25 + perturbation:
    rho = 2
else:
    rho = 1
ux = 0
uy = 0



