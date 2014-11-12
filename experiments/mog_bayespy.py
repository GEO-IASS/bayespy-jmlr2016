######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
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
#import matplotlib.pyplot as plt
#import time

#from bayespy.utils import misc
#import bayespy.plot as myplt
from bayespy import nodes
from bayespy.inference import VB

def mog_model(N, K, D):
    # N = number of data vectors
    # K = number of clusters
    # D = dimensionality
    
    # Construct the Gaussian mixture model

    # K prior weights (for components)
    alpha = nodes.Dirichlet(1e-3*np.ones(K),
                            name='alpha')
    # N K-dimensional cluster assignments (for data)
    z = nodes.Categorical(alpha,
                          plates=(N,),
                          name='z')
    # K D-dimensional component means
    X = nodes.Gaussian(np.zeros(D), 0.01*np.identity(D),
                       plates=(K,),
                       name='X')
    # K D-dimensional component covariances
    Lambda = nodes.Wishart(D, 0.01*np.identity(D),
                           plates=(K,),
                           name='Lambda')
    # N D-dimensional observation vectors
    Y = nodes.Mixture(z, nodes.Gaussian, X, Lambda, name='Y')

    z.initialize_from_random()

    return VB(Y, X, Lambda, z, alpha)


def generate_data(N):
    # Generate data
    #
    # This is really inefficient but easy to read. Efficiency isn't important in
    # generating the data.

    # True parameters
    mu1 = np.array([2.0, 3.0])
    mu2 = np.array([7.0, 5.0])
    Lambda1 = np.array([[3.0, 0.2],
                        [0.2, 2.0]])
    Lambda2 = np.array([[2.0, 0.4],
                        [0.4, 4.0]])
    pi = 0.6

    y = np.zeros((N,2))
    for n in range(N):
        if nodes.Bernoulli(pi).random():
            y[n] = Gaussian(mu1, Lambda1).random()
        else:
            y[n] = Gaussian(mu1, Lambda1).random()

    return y


def run(N=50, K=5, D=2):

    # Construct model
    Q = gaussianmix_model(N,K,D)

    # Observe data
    y = generate_data(N)
    Q['Y'].observe(y)

    # Run inference
    Q.update(repeat=30)

    ## # Run predictive model
    ## zh = nodes.Categorical(Q['alpha'], name='zh')
    ## Yh = nodes.Mixture(zh, nodes.Gaussian, Q['X'], Q['Lambda'], name='Yh')
    ## zh.update()

    ## # Plot predictive pdf
    ## N1 = 400
    ## N2 = 400
    ## x1 = np.linspace(-3, 15, N1)
    ## x2 = np.linspace(-3, 15, N2)
    ## xh = misc.grid(x1, x2)
    ## lpdf = Yh.integrated_logpdf_from_parents(xh, 0)
    ## pdf = np.reshape(np.exp(lpdf), (N2,N1))
    ## plt.clf()
    ## plt.contourf(x1, x2, pdf, 100)
    ## plt.scatter(y[:,0], y[:,1])
    ## print('integrated pdf:', np.sum(pdf)*(18*18)/(N1*N2))

    ## Q['X'].show()
    ## Q['alpha'].show()

    ## plt.show()

if __name__ == '__main__':
    run()

