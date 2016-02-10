import numpy

def upscale(a, n):
    """
    Upscales a k-dimensional array to size n^k
    :param a: numpy array
    :param n: new size, must be a multiple of a.shape
    :return:
    """

    while a.shape[0] < n:
        for axis in range(a.ndim):
            a = numpy.repeat(a, 2, axis)
    return a

def upscale_to_same_size(a, b):
    if a.shape[0] < b.shape[0]:
        a = upscale(a, b.shape[0])
    elif b.shape[0] < a.shape[0]:
        b = upscale(b, a.shape[0])
    return (a,b)