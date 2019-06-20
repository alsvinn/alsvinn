N = len(a)/8
a1 = a[:10]
a2 = a[10:20]
a3 = a[20:30]
a4 = a[30:40]
b1 = a[40:50]
b2 = a[50:60]
b3 = a[60:70]
b4 = a[70:80]


perturbation = 0.05
normalization1 = sum(a1)
if abs(normalization1) < 1e-10:
        normalization1 = 1
normalization2 = sum(a2)
if abs(normalization2) < 1e-10:
	normalization2 = 1

perturbation_upper = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(x+b1[i])) for i in range(len(a1))])/normalization1
perturbation_upper += perturbation*sum([a2[i]*cos(2*pi*(i+1)*(z+b2[i])) for i in range(len(a2))])/normalization1

perturbation_lower = perturbation*sum([a3[i]*cos(2*pi*(i+1)*(x+b3[i])) for i in range(len(a3))])/normalization2
perturbation_lower += perturbation*sum([a4[i]*cos(2*pi*(i+1)*(z+b4[i])) for i in range(len(a4))])/normalization2


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


