#!/bin/bash

for ((i=1; i<=4; i++))
do
    echo "Running iteration $i"
    python run.py -e 3dof-defend --n_cores 1 --n_episodes 100
done
