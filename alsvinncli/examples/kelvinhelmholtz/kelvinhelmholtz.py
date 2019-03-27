def init_global(rho, ux, uy, p, nx, ny, nz, ax, ay, az, bx, by, bz):
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

    x = linspace(ax, bx, nx)
    y = linspace(ay, by, ny)

    X, Y = meshgrid(x, y)
    X = X.T
    Y = Y.T
    perturbation_upper = 0.75 + perturbation*sum([a1[i]*cos(2*pi*(i+1)*(X+b1[i])) for i in range(len(a1))], 0)/normalization1
    perturbation_lower = 0.25 + perturbation*sum([a2[i]*cos(2*pi*(i+1)*(X+b2[i])) for i in range(len(a2))], 0)/normalization2
    middle = (Y < perturbation_upper)*(Y > perturbation_lower)

    rho[:, :, 0] = 2.0 * middle + 1.0*(1-middle)
    ux[:, :, 0] = -0.5*middle + 0.5*(1-middle)
    uy[:,:,0] = zeros_like(X)
    p[:,:,0] = 2.5*ones_like(X)


#

#
#
# perturbation = 0.01
# normalization1 = sum(a1)
# if abs(normalization1) < 1e-10:
#     normalization1 = 1
# normalization2 = sum(a2)
# if abs(normalization2) < 1e-10:
#     normalization2 = 1
#
# perturbation_upper = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(x+b1[i])) for i in range(len(a1))])/normalization1
# perturbation_lower = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(x+b2[i])) for i in range(len(a2))])/normalization2
#
# if y < 0.25 + perturbation_lower or y > 0.75 + perturbation_upper:
#     rho = 1
#     ux = 0.5
#     uy = 0
#     p = 2.5
# else:
#     rho = 2
#     ux = -0.5
#     uy = 0
#     p = 2.5
