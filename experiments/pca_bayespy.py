######################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# MIT License
######################################################################


import numpy as np

import matplotlib.pyplot as plt
import bayespy.plot as myplt

from bayespy.utils import misc
from bayespy.utils import random
from bayespy import nodes

from bayespy.inference.vmp.vmp import VB
from bayespy.inference.vmp import transformations

import bayespy.plot as bpplt

def model(M, N, D):
    # Construct the PCA model with ARD

    # ARD covariance
    alpha = nodes.Gamma(1e-2,
                        1e-2,
                        plates=(D,),
                        name='alpha')
    # Loadings
    W = nodes.GaussianARD(0,
                          alpha,
                          shape=(D,),
                          plates=(M,1),
                          name='W')

    # States
    X = nodes.GaussianARD(0,
                          1,
                          shape=(D,),
                          plates=(1,N),
                          name='X')

    # PCA
    F = nodes.SumMultiply('i,i', W, X,
                          name='F')

    # Noise
    tau = nodes.Gamma(1e-2, 1e-2,
                      name='tau')

    # Noisy observations
    Y = nodes.GaussianARD(F, tau,
                          name='Y')

    # Initialize some nodes randomly
    X.initialize_from_random()
    W.initialize_from_random()

    return VB(Y, F, W, X, tau, alpha)


def generate_data(M, N, D, seed=1):
    np.random.seed(seed)
    W = np.random.randn(M, D) / D
    X = np.random.randn(D, N)
    y = np.dot(W, X) + 0.3*np.random.randn(M, N)
    (u,s,v) = np.linalg.svd(y)
    np.savetxt('pca-data-%02d.csv' % seed, y, delimiter=',', fmt='%f')
    return y


def plot(seed=1, maxiter=None):
    """
    Show comparison plot
    """
    utils.plot('pca', seed, maxiter=maxiter)


def run(M=100, N=1000, D=10, seed=42, rotate=False, maxiter=200, debug=False, disable_broadcasting=False):

    # Generate data
    print("Generating data...")
    y = generate_data(M, N, np.ceil(D/2), seed=seed)

    # Construct model
    Q = model(M, N, D)

    # Observe data
    if disable_broadcasting:
        Q['Y'].observe(y, mask=np.ones((M,N), dtype=np.bool))
    else:
        Q['Y'].observe(y)

    # Run inference algorithm
    print("Running inference...")
    Q.update(repeat=maxiter, tol=0)

    v = np.array([Q.L[:(Q.iter+1)], Q.cputime[:(Q.iter+1)]]).T
    np.savetxt("pca-results-%02d-bayespy.csv" % seed, v, delimiter=",")


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["m=",
                                    "n=",
                                    "d=",
                                    "seed=",
                                    "maxiter=",
                                    "disable-broadcasting",
                                    "debug",
                                    "rotate"])
    except getopt.GetoptError:
        print('python pca_bayespy.py <options>')
        print('--m=<INT>        Dimensionality of data vectors')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--rotate         Apply speed-up rotations')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        print('--disable-broadcasting Do not utilize broadcasting')
        print('--debug          Check that the rotations are implemented correctly')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--rotate":
            kwargs["rotate"] = True
        elif opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--debug":
            kwargs["debug"] = True
        elif opt == "--disable-broadcasting":
            kwargs["disable_broadcasting"] = True
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--m",):
            kwargs["M"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        elif opt in ("--d",):
            kwargs["D"] = int(arg)

    run(**kwargs)
    plt.show()

