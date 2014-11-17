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


def load_data_iris():
    return np.genfromtxt('mog_data/bezdekIris.data',
                         usecols=tuple(range(0,4)),
                         delimiter=',')


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
            y[n] = nodes.Gaussian(mu1, Lambda1).random()
        else:
            y[n] = nodes.Gaussian(mu2, Lambda2).random()

    return y


def run(K=10):

    # Get data
    #y = generate_data(N)
    y = load_data_iris()

    # Construct model
    (N, D) = np.shape(y)
    Q = mog_model(N,K,D)

    # Observe data
    Q['Y'].observe(y)

    # Run inference
    Q.update(repeat=50)

    print(Q['alpha'].u[0])


if __name__ == '__main__':
    run()

