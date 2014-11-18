import numpy as np
import pymc

# Number of samples
N = 100

# Sample dimensionality
M = 10

# Latent space dimensionality
D = 3

def collapse(l):
    """Collapse a list of lists into a list."""
    return [item for sublist in l for item in sublist]

# Just some dummy data
y = np.random.randn(M, N)

# ARD parameters
alpha = pymc.Gamma('alpha', alpha=1e-2, beta=1e-2, size=D)
@pymc.deterministic(dtype=float)
def Alpha(a=alpha):
    return np.diag(a)

# Loading matrix
W = [pymc.MvNormal('W%i'%i, np.zeros(D), Alpha)
     for i in range(M)]

# Latent states
X = [pymc.MvNormal('z%i'%i, np.zeros(D), np.eye(D))
     for i in range(N)]

# Noiseless values
bias = 0 # bias term, ignore it for simplicity
F = [[pymc.Lambda('f%i%i'%(i,j),
                  lambda W=W, X=X: bias + np.dot(np.array(W[i]),
                                                 np.array(X[j])))
      for j in range(N)]
     for i in range(M)]

# Noise
tau = pymc.Gamma('tau', alpha=1e-2, beta=1e-2)

# Noisy observations
Y = [[pymc.Normal('Obs%i'%i, F[i][j], tau, value=y[i,j], observed=True)
      for j in range(N)]
     for i in range(M)]

# Construct model
model = pymc.Model([tau, alpha] + collapse(Y) + W + X)
mcmc = pymc.MCMC(model,
                 db='sqlite',
                 dbname='pca_pymc.db')

# Run inference
print("PCA model constructed.")
print("Run inference using MCMC...")
mcmc.sample(iter=10, burn=0, thin=1)
