epsilon = 6e-2
xc = x - 0.5
yc = y - 0.5
phi = atan2(yc, xc) if abs(xc) > 0 else 0

if phi < 0:
    phi += 2*pi

N = len(a)/2

a1 = a[:10]

b1 = a[10:20]

normalization = sum(a1)

if abs(normalization) < 1e-8:
    normalization = N

perturbation = epsilon * sum([a1[n] * cos(phi+2*pi*b1[n]) for n in range(N)]) / normalization

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



