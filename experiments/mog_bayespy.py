######################################################################
# Copyright (C) 2011-2015 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

import os

import numpy as np
import matplotlib.pyplot as plt
#import time

#from bayespy.utils import misc
#import bayespy.plot as myplt
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
    alpha = nodes.Dirichlet(1.0*np.ones(K),
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

    return VB(Y, mu, Lambda, z, alpha)


## def load_data_iris():
##     return np.genfromtxt('mog_data/bezdekIris.data',
##                          usecols=tuple(range(0,4)),
##                          delimiter=',')


def generate_data(N, D, K, seed=1):
    """
    Generate data from a mixture of Gaussians model
    """

    np.random.seed(seed)

    mu = np.random.randn(K, D)
    # Lambda is actually precision matrix (inverse covariance)
    Lambda = random.covariance(D, size=K)
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

    result_in = np.loadtxt('mog-results-%02d-infernet.csv' % seed,
                           delimiter=',')

    result_bp = np.loadtxt('mog-results-%02d-bayespy.csv' % seed,
                           delimiter=',')

    if maxiter is not None:
        result_in = result_in[:maxiter,:]
        result_bp = result_bp[:maxiter,:]

    loglike_in = result_in[:,0]
    loglike_bp = result_bp[:,0]
    cputime_in = np.cumsum(result_in[:,1]) / 1000
    cputime_bp = np.cumsum(result_bp[:,1])

    # Show curves
    plt.plot(cputime_in, loglike_in, 'k--')
    plt.plot(cputime_bp, loglike_bp, 'r-')

    # Show markers for every 10th iteration
    plt.plot(cputime_in[9::10], loglike_in[9::10], '*', markeredgecolor='k', markerfacecolor='k')
    plt.plot(cputime_bp[9::10], loglike_bp[9::10], '*', markeredgecolor='r', markerfacecolor='r')

    plt.xlabel('CPU time (seconds)')
    plt.ylabel('VB lower bound')

    plt.legend(['Infer.NET', 'BayesPy'], loc='lower right')


def run(N=2000, D=3, K=20, seed=1, maxiter=200):

    # Get data
    print("Generating data...")
    y = generate_data(N, D, K, seed=seed)

    # Construct model
    Q = mog_model(N, 2*K, D)

    # Observe data
    Q['Y'].observe(y)

    # Run inference
    print("Running inference...")
    Q.update(repeat=maxiter, tol=0)

    Q['alpha'].show()

    v = np.array([Q.L[:(Q.iter+1)], Q.cputime[:(Q.iter+1)]]).T
    np.savetxt("mog-results-%02d-bayespy.csv" % seed, v, delimiter=",")

    # Run Infer.NET
    os.system('make && mono mog_infernet.exe %d %d %d' % (2*K, seed, maxiter))


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["n=",
                                    "d=",
                                    "k=",
                                    "maxiter=",
                                    "seed="])
    except getopt.GetoptError:
        print('python demo_pca.py <options>')
        print('--n=<INT>        Number of data vectors')
        print('--d=<INT>        Dimensionality of the latent vectors in the model')
        print('--k=<INT>        Dimensionality of the true latent vectors')
        print('--maxiter=<INT>  Maximum number of VB iterations')
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
        elif opt in ("--maxiter",):
            kwargs["maxiter"] = int(arg)

    run(**kwargs)


