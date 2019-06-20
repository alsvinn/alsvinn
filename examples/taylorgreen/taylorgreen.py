
GAMMA=5.0/3.0

k = 1
u0 = 1

p_inf = 100

ux = u0 * sin(2*pi*k*x)*cos(2*pi*k*y)*cos(2*pi*k*z)
uy = -u0 * cos(2*pi*k*x)*sin(2*pi*k*y)*cos(2*pi*k*z)
uz = 0

rho = 1
p = p_inf + 1.0/16.0*rho*u0**2 * (2 + cos(4*pi*k*z))*(cos(4*k*pi*x) + cos(4*pi*k*z))

