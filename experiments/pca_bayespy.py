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

    if False:
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
    elif False:

        # Full covariance
        alpha = nodes.Wishart(D,
                              D*np.identity(D),
                              name='alpha')

        # Loadings
        W = nodes.Gaussian(np.zeros(D),
                           alpha,
                           plates=(M,1),
                           name='W')
    else:
        # Constant covariance

        # Dummy node
        alpha = nodes.Bernoulli(0.5, name='alpha')

        # Loadings
        W = nodes.GaussianARD(0,
                              1,
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
    W = np.random.randn(M, D)
    X = np.random.randn(D, N)
    y = np.dot(W, X) + 0.1*np.random.randn(M, N)
    np.savetxt('pca-data-%02d.csv' % seed, y, delimiter=',', fmt='%f')
    return y




@bpplt.interactive
def run(M=100, N=1000, D=10, seed=42, rotate=False, maxiter=200, debug=False):

    # Generate data
    print("Generating data...")
    y = generate_data(M, N, D, seed=seed)

    # Construct model
    Q = model(M, N, 2*D)

    # Observe data
    Q['Y'].observe(y)

    # Run inference algorithm
    if rotate:
        # Use rotations to speed up learning
        rotW = transformations.RotateGaussianARD(W, alpha)
        rotX = transformations.RotateGaussianARD(X)
        R = transformations.RotationOptimizer(rotW, rotX, D)
        Q.set_callback(R.rotate)
            
    # Use standard VB-EM alone
    print("Running inference...")
    Q.update(repeat=maxiter, tol=0)

    Q['W'].show()
    Q['X'].show()
    Q['tau'].show()


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

