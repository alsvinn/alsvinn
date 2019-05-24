N = len(X)
u = sum([X[k-1]*sin((k - 0.5) * pi * x) / ((k - 0.5) * pi) for k in range(1,N+1)])
u += 5
u /= 20
