######################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# MIT License
######################################################################


import numpy as np
import matplotlib.pyplot as plt

def plot(method, seed, maxiter=None):
    """
    Show comparison plot
    """

    result_in = np.loadtxt('%s-results-%02d-infernet.csv' % (method, seed),
                           delimiter=',')

    result_bp = np.loadtxt('%s-results-%02d-bayespy.csv' % (method, seed),
                           delimiter=',')

    if maxiter is not None:
        result_in = result_in[:maxiter,:]
        result_bp = result_bp[:maxiter,:]

    loglike_in = result_in[:,0]
    loglike_bp = result_bp[:,0]
    cputime_in = np.cumsum(result_in[:,1])
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

    print("Method={0}, seed={1}, package={2}, CPU time={3}".format(method,
                                                                   seed,
                                                                   "bayespy",
                                                                   np.mean(result_bp[:,1])))
    print("Method={0}, seed={1}, package={2}, CPU time={3}".format(method,
                                                                   seed,
                                                                   "infernet",
                                                                   np.mean(result_in[:,1])))


