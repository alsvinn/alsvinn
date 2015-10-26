import alsvinn.data
import numpy
class LpNorm(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, a, b):
        """
        Computes ||a-b||_p. Upscales the data if necessary
        :param a: array (n-dimensional)
        :param b: array (n-dimension)
        :return: ||a-b||_p.
        """

        (a,b) = alsvinn.data.upscale_to_same_size(a, b)
        area = numpy.prod(a.shape)
        return numpy.sum(abs(a-b)**self.p) / area