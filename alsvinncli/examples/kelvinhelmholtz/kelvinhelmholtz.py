N = len(a)/4
a1 = a[:10]
a2 = a[10:20]
b1 = a[20:40]
b2 = a[30:40]


perturbation = 0.05
normalization1 = sum(a1)
if abs(normalization1) < 1e-10:
	normalization1 = 1
normalization2 = sum(a2)
if abs(normalization2) < 1e-10:
	normalization2 = 1

perturbation_upper = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(x+b1[i])) for i in range(len(a1))])/normalization1
perturbation_lower = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(x+b2[i])) for i in range(len(a2))])/normalization2

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


