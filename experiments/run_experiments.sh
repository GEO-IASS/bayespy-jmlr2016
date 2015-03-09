#!/bin/bash

# PCA experiments
python pca_bayespy.py --n=500 --m=20 --d=10 --maxiter=200 --seed=1
# make && mono pca_infernet.exe 10 1 200 0
make && mono pca_infernet.exe 10 1 200 1
python pca_bayespy.py --n=2000 --m=100 --d=40 --maxiter=200 --seed=2
# make && mono pca_infernet.exe 40 2 200 0
make && mono pca_infernet.exe 40 2 200 1

# MoG experiments
python mog_bayespy.py --n=200 --k=10 --d=2 --maxiter=200 --seed=1 --spread=3.0
make && mono mog_infernet.exe 10 1 200
python mog_bayespy.py --n=2000 --k=40 --d=10 --maxiter=200 --seed=2 --spread=1.0
make && mono mog_infernet.exe 40 2 200

