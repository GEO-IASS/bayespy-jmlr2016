######################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# MIT License
######################################################################

import os

import numpy as np
import matplotlib.pyplot as plt

from bayespy import nodes
from bayespy.inference import VB
from bayespy.utils import random


def mog_model(N, K, D):
    """
    Construct a mixture of Gaussians model
    """
    # N = number of data vectors
    # K = number of clusters
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # K prior weights (for components)
    alpha = nodes.Dirichlet(0.1*np.ones(K),
                            name='alpha')
    # N K-dimensional cluster assignments (for data)
    z = nodes.Categorical(alpha,
                          plates=(N,),
                          name='z')
    # K D-dimensional component means
    mu = nodes.Gaussian(np.zeros(D), 0.01*np.identity(D),
                        plates=(K,),
                        name='mu')
    # K D-dimensional component covariances
    Lambda = nodes.Wishart(D, D*np.identity(D),
                           plates=(K,),
                           name='Lambda')
    # N D-dimensional observation vectors
    Y = nodes.Mixture(z, nodes.Gaussian, mu, Lambda, name='Y')

    # Break symmetry
    z.initialize_from_random()

    return VB(Y, mu, z, Lambda, alpha)


def generate_data(N, D, K, seed=1, spread=3):
    """
    Generate data from a mixture of Gaussians model
    """

    np.random.seed(seed)

    mu = spread*np.random.randn(K, D)
    # Lambda is actually precision matrix (inverse covariance)
    Lambda = random.covariance(D, size=K, nu=2*D)
    pi = random.dirichlet(5*np.ones(K))

    y = np.zeros((N,D))

    for n in range(N):
        ind = nodes.Categorical(pi).random()
        y[n] = nodes.Gaussian(mu[ind], Lambda[ind]).random()

    np.savetxt('mog-data-%02d.csv' % seed, y, delimiter=',', fmt='%f')

    return y


def plot(seed=1, maxiter=None):
    """
    Show comparison plot
    """
    utils.plot('mog', seed, maxiter=maxiter)


def run(N=2000, D=3, K=20, seed=1, maxiter=200, spread=3.0):

    # Get data
    print("Generating data...")
    y = generate_data(N, D, int(np.ceil(K/2)), seed=seed, spread=spread)

    # Construct model
    Q = mog_model(N, K, D)

    # Observe data
    Q['Y'].observe(y)

    # Run inference
    print("Running inference...")
    Q.update(repeat=maxiter, tol=-1e-6)

    v = np.array([Q.L[:(Q.iter+1)], Q.cputime[:(Q.iter+1)]]).T
    np.savetxt("mog-results-%02d-bayespy.csv" % seed, v, delimiter=",")


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "d=",
                                    "k=",
                                    "maxiter=",
                                    "spread=",
                                    "seed="])
    except getopt.GetoptError:
        print('python mog_bayespy.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the true latent vectors')
        print('--maxiter=<INT>  Maximum number of VB iterations')
        print('--spread=<NUM>   Std/spread for true cluster means')
        print('--seed=<INT>     Seed (integer) for the random number generator')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--n",):
            kwargs["N"] = int(arg)
        elif opt in ("--d",):
            kwargs["D"] = int(arg)
        elif opt in ("--k",):
            kwargs["K"] = int(arg)
        elif opt in ("--spread",):
            kwargs["spread"] = float(arg)
        elif opt in ("--maxiter",):
            kwargs["maxiter"] = int(arg)

    run(**kwargs)


