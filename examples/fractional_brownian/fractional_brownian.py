import scipy.stats
# This is found in the paper
# Brouste, A., Istas, J., & Lambert-Lacroix, S. (2007).
#  On Fractional Gaussian Random Fields Simulations. Journal of Statistical Software, 23(1), 1–23.
#  http://doi.org/http://dx.doi.org/10.18637/jss.v023.i01
# in the section 2.3
def init_global(u, nx, ny, nz, ax, ay, az, bx, by, bz):
    H = hurst

    # this is the inverse cdf of the normal distribution
    # X is uniform, so Y will be Gaussian
    Y = scipy.stats.norm.ppf(X)
    # Uses fbmpy, available from https://github.com/kjetil-lye/fractional_brownian_motion
    B_max = max(abs(fbmpy.fractional_brownian_bridge_1d(H, Y.shape[0], Y)))
    B = fbmpy.fractional_brownian_bridge_1d(H, nx, Y)/B_max

    u[:,0,0]= B[:-1]



