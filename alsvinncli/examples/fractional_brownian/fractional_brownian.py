# This is found in the paper
# Brouste, A., Istas, J., & Lambert-Lacroix, S. (2007).
#  On Fractional Gaussian Random Fields Simulations. Journal of Statistical Software, 23(1), 1–23.
#  http://doi.org/http://dx.doi.org/10.18637/jss.v023.i01
# in the section 2.3
def init_global(u, nx, ny, nz, ax, ay, az, bx, by, bz):
    H = 0.5
    M = u.shape[0]
    N = M+1

    K = log2(M)
    B = zeros(N)
    B[-1] = X[len(X)-1]
    n=1
    random_counter = 0
    for k in range(int(K),0,-1):
        for j in range(0, int(M/2**k)):

            index = 2**(k-1)*(2*j+1)
            left = index - 2**(k-1)
            right = index + 2**(k-1)
            B[index] = 0.5*(B[left]+B[right])+sqrt((1-2**(2*H-2))/2**(2*n*H))*X[random_counter]
            random_counter += 1
        n+=1

    u[:,0,0]= B[:-1]



