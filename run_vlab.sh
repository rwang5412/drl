#PBS -l walltime=768:00:00 -lselect=1:ncpus=112 -lplace=excl

PROJECT_DIR=~/roadrunner_refactor/
cd $PROJECT_DIR
source /export/software/anaconda3/bin/activate
conda activate mjenv

python train.py ppo \
--env-name 'CassieEnvClock' --simulator_type 'mujoco' \
--policy_rate 50 \
--reward_name 'locomotion_linear_clock_reward' --clock_type 'linear' \
--run_name 'init_test' \
--layers 64,64 --discount 0.95 \
--std 0.13 --mirror 1 --batch_size 32 --num_steps 50000 --a_lr 0.0003 --c_lr 0.0003 --epochs 5 \
--traj_len 300 --timesteps 4000000000 --workers 56 --arch ff --wandb
