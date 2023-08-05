#!/bin/bash
#PBS -l walltime=768:00:00 -lselect=1:ncpus=160 -lplace=excl

PROJECT_DIR=~/roadrunner_refactor/
cd $PROJECT_DIR
source activate env1

python test.py --train
