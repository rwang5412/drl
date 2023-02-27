#PBS -l walltime=768:00:00 -lselect=1:ncpus=112 -lplace=excl

PROJECT_DIR=~/roadrunner_refactor/
cd $PROJECT_DIR
source /export/software/anaconda3/bin/activate
conda activate mjenv

python train.py ppo \
--env-name 'CassieStone' \
--run-name 'init_test_stone' \
--layers 64,64 --discount 0.95 \
--std 0.13 --mirror 1 --batch-size 32 --num-steps 50000 --a-lr 0.0003 --c-lr 0.0003 --epochs 5 \
--traj-len 300 --timesteps 4000000000 --workers 56 --arch lstm --wandb
