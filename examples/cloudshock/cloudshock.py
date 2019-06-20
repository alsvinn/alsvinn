N = len(a)/4
a1 = a[:10]
a2 = a[10:20]
b1 = a[20:40]
b2 = a[30:40]


perturbation = 0.1
normalization1 = sum(a1)
if abs(normalization1) < 1e-10:
    normalization1 = 1
normalization2 = sum(a2)
if abs(normalization2) < 1e-10:
    normalization2 = 1
#
perturbation_x = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(x+b1[i])) for i in range(len(a1))])/normalization1
perturbation_r = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(x+b2[i])) for i in range(len(a2))])/normalization2
#

if x < 0.05 + perturbation_x:
    rho = 3.86859
    ux = 11.2536;
    p = 167.345
else:
    r = (x-0.25)**2 + (y-0.5)**2
    phi = (x-0.25) / sqrt(r)
    r_max = 0.13**2
    if r <= r_max + perturbation_r:
        rho = 10.0
    else:
        rho = 1.0
    p = 1.0
