with open("progress.txt","a") as f:
    f.write("%f\n"%x)

N = len(X)
K = arange(1,N+1,1)
u = sum(X*sin((K-0.5)*pi*x)/((K-0.5)*pi))

# sum([X[k-1]*sin((k - 0.5) * pi * x) / ((k - 0.5) * pi) for k in range(1,N+1)])

