#!/bin/bash

# MoG experiments
python mog_bayespy.py --n=200 --k=10 --d=2 --maxiter=200 --seed=1
make && mono mog_infernet.exe 10 1 200
python mog_bayespy.py --n=2000 --k=40 --d=5 --maxiter=200 --seed=2
make && mono mog_infernet.exe 40 2 200
