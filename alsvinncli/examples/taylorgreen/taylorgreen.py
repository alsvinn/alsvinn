N = len(a)/4
a1 = a[:10]
a2 = a[10:20]
b1 = a[20:40]
b2 = a[30:40]
GAMMA=5.0/3.0

k = 1
u0 = 1

p_inf = 100
perturbation=0.01
perturbation_x = perturbation*sum([a1[i]*cos(2*pi*(i+1)*(y+b1[i])) for i in range(len(a1))])
perturbation_y = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(x+b2[i])) for i in range(len(a2))])
perturbation_z = perturbation*sum([a2[i]*cos(2*pi*(i+1)*(y+b2[i])) for i in range(len(a2))])

ux = u0 * sin(2*pi*k*(x+perturbation_x))*cos(2*pi*k*(y+perturbation_y))*cos(2*pi*k*(z+perturbation_z))
uy = -u0 * cos(2*pi*k*x)*sin(2*pi*k*y)*cos(2*pi*k*z)
uz = 0

rho = 1
p = p_inf + 1.0/16.0*rho*u0**2 * (2 + cos(4*pi*k*(z+perturbation_x)))*(cos(4*k*pi*(x+perturbation_x)) + cos(4*pi*k*(z+perturbation_z)))

ux = u0 * sin(2*pi*k*x)*cos(2*pi*k*y)*cos(2*pi*k*z)
uy = -u0 * cos(2*pi*k*x)*sin(2*pi*k*y)*cos(2*pi*k*z)
uz = 0

rho = 1
p = p_inf + 1.0/16.0*rho*u0**2 * (2 + cos(4*pi*k*z))*(cos(4*k*pi*x) + cos(4*pi*k*z))

