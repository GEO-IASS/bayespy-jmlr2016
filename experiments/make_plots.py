# Copyright (c) 2015 Jaakko Luttinen

from utils import plot

import matplotlib.pyplot as plt

if __name__ == "__main__":

    plt.figure()
    plot("pca", 1, maxiter=200)
    plt.xlim(0, 6)
    plt.ylim(-7500, -4500)
    plt.savefig("fig_pca_01.pdf", frameon=False)
    
    plt.figure()
    plot("pca", 2, maxiter=200)
    plt.xlim(0, 70)
    plt.ylim(-120000, -70000)
    plt.savefig("fig_pca_02.pdf", frameon=False)
    
    plt.figure()
    plot("mog", 1, maxiter=200)
    plt.xlim(0, 3)
    plt.ylim(-1000, -750)
    plt.savefig("fig_mog_01.pdf", frameon=False)

    plt.figure()
    plot("mog", 2, maxiter=200)
    plt.xlim(0, 150)
    plt.ylim(-40000, -32000)
    plt.savefig("fig_mog_02.pdf", frameon=False)

    plt.show()
