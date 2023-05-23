#PBS -l walltime=768:00:00 -lselect=1:ncpus=160 -lplace=excl

source activate rr_refactor
cd ~/roadrunner_refactor
python ~/roadrunner_refactor/run_ppo.py --seed=$s
