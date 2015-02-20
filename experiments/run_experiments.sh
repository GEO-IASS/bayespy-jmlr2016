#!/bin/bash

# MoG experiments (smaller and larger)
python mog_bayespy.py --n=200 --k=5 --d=2 --maxiter=200 --seed=1
python mog_bayespy.py --n=2000 --k=20 --d=5 --maxiter=200 --seed=2
